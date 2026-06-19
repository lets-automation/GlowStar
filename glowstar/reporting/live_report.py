"""LIVE Excel report — everything pulled in real time, nothing hardcoded.

Pipeline for this report:
  1. Pull the live inventory+sales book from the Channel Partner API and bank an
     immutable snapshot.
  2. Out-of-time split on the live data; train the engine on the earlier sales.
  3. For the held-out report stones, pull LIVE Uni comparables per segment, clean
     them (authenticity pipeline), and build a LIVE market table — the displayed
     price is anchored to current market, the comparables shown are real-time.
  4. Write the Excel: real actual sale vs live-anchored suggestion, the live
     market median + comparable count per stone, the internal trend (computed
     from live sales), and a provenance sheet. No seeded macro figures.

Run:  python -m glowstar.reporting.live_report            # ~100 stones (live calls)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR, SETTINGS
from ..data.loaders import load_records, sold_stones
from ..ingestion import channel_partner
from ..ingestion.snapshots import save_snapshot
from ..market.live import LiveMarket
from ..models.engine import PricingEngine, EngineConfig
from ..narration.narrate import template_narration
from ..validation.backtest import time_split

log = logging.getLogger(__name__)
OUT_PATH = ARTIFACTS_DIR / "GlowStar_Pricing_Report_LIVE.xlsx"


def build(n: int = 100, split_date: str | None = None, out: Path | None = None,
          seed: int = 42) -> Path:
    split_date = split_date or SETTINGS.backtest_split_date
    out = out or OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. LIVE records.
    log.info("Pulling live inventory+sales from Channel Partner API ...")
    records = channel_partner.get_all_records()
    snap = save_snapshot(records, source="channel_partner")
    df, rep = load_records(snap.path)
    log.info("LIVE %s", rep.summary())

    sold = sold_stones(df, drop_outliers=True)
    train, test, info = time_split(sold, split_date)
    if len(test) < n:
        n = len(test)

    # 2. Train (market anchor calibrated on the banked full-market snapshot).
    log.info("Training engine on %s sold ...", f"{len(train):,}")
    eng = PricingEngine(EngineConfig(split_date=split_date)).fit(train)
    log.info("Engine trained.")

    # 3. LIVE Uni comparables for the report stones -> live market table.
    sample = test.sample(n, random_state=seed).reset_index(drop=True)
    log.info("Pulling LIVE Uni comparables for %s stones (cached per segment) ...", n)
    lm = LiveMarket()
    live_tables = lm.build_tables(sample, base=eng.tables)
    eng.tables = live_tables                      # price against live market
    log.info("Live Uni calls made: %s (unique segments)", lm.calls)

    sugg = eng.predict(sample)

    # 4. Build rows.
    from ..market.live import no_bgm_median
    rows, live_meds = [], []
    for st, s in zip(sample.itertuples(), sugg):
        comp = lm.comparables(st.Shape_full, st.Weight, st.Color, st.Clarity, st.Lab, st.Fluorescence)
        live_med = comp.report.median_discount if comp else None
        live_n = comp.report.n_used if comp else 0
        dup_rate = comp.report.duplicate_rate if comp else None
        nobgm_med, nobgm_n = no_bgm_median(comp.stones) if comp else (None, 0)
        bgm_free_share = (nobgm_n / live_n) if (comp and live_n) else None
        if live_med is not None:
            live_meds.append(live_med)
        actual = float(st.FDiscount)
        facts = {"suggested_discount": s.suggested_discount, "suggested_ppc": s.suggested_ppc,
                 "suggested_net": s.suggested_net, "ci_discount_low": s.ci_discount_low,
                 "ci_discount_high": s.ci_discount_high, "comparable_count": s.comparable_count,
                 "market_median_discount": s.market_median_discount, "method": s.method,
                 "flags": s.flags, "coverage_pct": int(eng.cfg.coverage * 100)}
        rows.append({
            "StoneId": st.StoneId, "Shape": st.Shape_full, "Weight(ct)": st.Weight,
            "Color": st.Color, "Clarity": st.Clarity, "Cut(CPS)": st.CPS,
            "Fluorescence": st.Fluorescence, "Lab": st.Lab, "Rap($/ct)": st.Rap,
            "ACTUAL Disc%": round(actual, 2), "ACTUAL Net$": round(float(st.FNetAmount), 2),
            "SUGGESTED Disc%": s.suggested_discount, "SUGGESTED Net$": s.suggested_net,
            "Disc Error(pts)": round(s.suggested_discount - actual, 2),
            "Abs Disc Err": round(abs(s.suggested_discount - actual), 2),
            "CI Low%": s.ci_discount_low, "CI High%": s.ci_discount_high,
            "Actual in band?": "yes" if s.ci_discount_low <= actual <= s.ci_discount_high else "no",
            "Method": s.method,
            "Market median used Disc%": s.market_median_discount,
            "LIVE no-BGM median Disc%": None if nobgm_med is None else round(nobgm_med, 2),
            "LIVE BGM-free %": None if bgm_free_share is None else f"{bgm_free_share:.0%}",
            "Market source": "live" if live_med is not None else "banked-fallback",
            "LIVE comps (n)": live_n,
            "LIVE dup rate removed": None if dup_rate is None else f"{dup_rate:.0%}",
            # Transparent build-up of the suggested discount:
            "BGM deduction(pts)": s.bgm_deduction_pts,
            "Trend adj(pts)": s.trend_shift_pts,
            "Feedback adj(pts)": s.feedback_correction_pts,
            "BGM state": s.bgm_state, "Market direction (from sales)": s.market_direction,
            "Flags": ", ".join(s.flags),
            "Why (explanation)": template_narration(facts),
        })
    res = pd.DataFrame(rows)

    # 5. Sheets.
    err = res["Abs Disc Err"].to_numpy()
    summary = pd.DataFrame([
        ["Data source", "LIVE Channel Partner API pull (banked this run)"],
        ["Live records pulled", f"{len(records):,}"],
        ["Trained on (held-out split)", f"{info.n_train:,} sold before {split_date}"],
        ["Scored (held-out)", f"{len(res)} sold on/after {split_date}"],
        ["Live Uni comparable calls", f"{lm.calls} unique segments"],
        ["Rows priced on LIVE market", f"{(res['Market source'] == 'live').sum()} of {len(res)} "
         f"({(res['Market source'] == 'live').mean():.0%}); rest fall back to banked market data"],
        ["MAE (disc pts) on sample", round(float(err.mean()), 2)],
        ["Within +/-5 pts", f"{np.mean(err <= 5):.0%}"],
        ["Median $ error", round(float((res['SUGGESTED Net$'] - res['ACTUAL Net$']).abs().median()), 0)],
        ["Actual inside band", f"{(res['Actual in band?'] == 'yes').mean():.0%}"],
        ["Avg live market median (sampled)", round(float(np.mean(live_meds)), 2) if live_meds else None],
        ["Internal trend direction (from live sales)", eng.index.as_dict()["direction"]],
    ], columns=["Field", "Value"])

    trend = pd.DataFrame([{"Month": m, "Quality-adjusted index (pts)": v}
                          for m, v in eng.index.as_dict()["monthly_index"].items()])

    provenance = pd.DataFrame([
        ("Inventory/sales records", "LIVE — pulled from Channel Partner API this run, banked as an immutable snapshot."),
        ("Suggested price", "LIVE model output — trained on live sales, anchored to LIVE Uni market medians per stone."),
        ("LIVE market median / comps", "LIVE — fresh Uni export-report pull per segment, deduped & cleaned (authenticity pipeline)."),
        ("Market direction", "REAL — computed from the client's own live sales (quality-adjusted index)."),
        ("Macro figures (RAPI etc.)", "NOT INCLUDED here — this report contains only live/real-data values, no seeded macro."),
        ("Actual sale values", "REAL — the held-out stone's true realized sale from the live book."),
    ], columns=["Field", "Source (all live / real)"])

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Overview (LIVE)", index=False)
        res.to_excel(xl, sheet_name="Pricing Results (LIVE)", index=False)
        trend.to_excel(xl, sheet_name="Market Trend (live sales)", index=False)
        provenance.to_excel(xl, sheet_name="Provenance", index=False)
        for ws in xl.book.worksheets:
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(58, max(10, w + 2))
    log.info("Wrote LIVE report -> %s (MAE %.2f on %s stones)", out, err.mean(), len(res))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Wrote", build())


if __name__ == "__main__":
    main()
