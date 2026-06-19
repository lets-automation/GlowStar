"""Excel report: 100 held-out stones — actual vs suggested, why, market data.

Honesty by construction: the 100 stones are drawn from the OUT-OF-TIME test set
(sales after the split). The engine is trained ONLY on earlier sales, so it has
never seen these stones' outcomes — "ActualDiscount" vs "SuggestedDiscount" is a
fair comparison, not the model reading the answer.

Sheets:
  1. Pricing Results — one row per stone: real sale, system suggestion, error,
     the reasons (method, comparables, market level, BGM, trend), explanation.
  2. Accuracy Summary — metrics over these 100 + per shape.
  3. Market Research — the internal trend index + external macro signals (with
     sources) + the internal-vs-macro cross-check.
  4. Legend & Data Honesty — what every column means and, explicitly, what is
     computed-from-real-data vs seeded/hardcoded vs not-yet-live.

Run:  python -m glowstar.reporting.excel_report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR, SETTINGS
from ..data.loaders import load_records, sold_stones
from ..market.context import current_context
from ..models.engine import PricingEngine, EngineConfig
from ..narration.narrate import template_narration
from ..validation.backtest import time_split

OUT_PATH = ARTIFACTS_DIR / "GlowStar_Pricing_Report.xlsx"


def _results_frame(n: int, split_date: str, seed: int = 42):
    df, _ = load_records()
    sold = sold_stones(df, drop_outliers=True)
    train, test, info = time_split(sold, split_date)
    eng = PricingEngine(EngineConfig(split_date=split_date)).fit(train)

    sample = test.sample(min(n, len(test)), random_state=seed).reset_index(drop=True)
    sugg = eng.predict(sample)
    ctx = current_context()

    rows = []
    for st, s in zip(sample.itertuples(), sugg):
        facts = {
            "suggested_discount": s.suggested_discount, "suggested_ppc": s.suggested_ppc,
            "suggested_net": s.suggested_net, "ci_discount_low": s.ci_discount_low,
            "ci_discount_high": s.ci_discount_high, "comparable_count": s.comparable_count,
            "market_median_discount": s.market_median_discount, "method": s.method,
            "flags": s.flags, "coverage_pct": int(eng.cfg.coverage * 100),
        }
        actual_disc = float(st.FDiscount)
        actual_net = float(st.FNetAmount)
        within = s.ci_discount_low <= actual_disc <= s.ci_discount_high
        rows.append({
            "StoneId": st.StoneId, "Shape": st.Shape_full, "Weight(ct)": st.Weight,
            "Color": st.Color, "Clarity": st.Clarity, "Cut(CPS)": st.CPS,
            "Fluorescence": st.Fluorescence, "Lab": st.Lab, "Location": st.Location,
            "Rap($/ct)": st.Rap,
            "ACTUAL Disc%": round(actual_disc, 2), "ACTUAL Net$": round(actual_net, 2),
            "SUGGESTED Disc%": s.suggested_discount, "SUGGESTED Net$": s.suggested_net,
            "Disc Error(pts)": round(s.suggested_discount - actual_disc, 2),
            "Abs Disc Err": round(abs(s.suggested_discount - actual_disc), 2),
            "Net Error$": round(s.suggested_net - actual_net, 2),
            "CI Low%": s.ci_discount_low, "CI High%": s.ci_discount_high,
            "Actual in band?": "yes" if within else "no",
            "Method": s.method, "Market comps (n)": s.comparable_count,
            "Market median Disc%": s.market_median_discount,
            "BGM state": s.bgm_state, "BGM deduction(pts)": s.bgm_deduction_pts,
            "Trend shift(pts)": s.trend_shift_pts,
            "Feedback corr(pts)": s.feedback_correction_pts,
            "Market direction": s.market_direction,
            "Flags": ", ".join(s.flags),
            "Why (explanation)": template_narration(facts),
        })
    res = pd.DataFrame(rows)
    return res, info, ctx, eng


def _accuracy_summary(res: pd.DataFrame) -> pd.DataFrame:
    err = res["Abs Disc Err"].to_numpy()
    overall = {
        "Scope": f"{len(res)} held-out stones (out-of-time)",
        "MAE (disc pts)": round(float(err.mean()), 2),
        "Median abs err (pts)": round(float(np.median(err)), 2),
        "Within +/-3 pts": f"{np.mean(err <= 3):.0%}",
        "Within +/-5 pts": f"{np.mean(err <= 5):.0%}",
        "Median $ error": round(float(res["Net Error$"].abs().median()), 0),
        "Actual inside band": f"{(res['Actual in band?'] == 'yes').mean():.0%}",
    }
    by_shape = (res.assign(ae=res["Abs Disc Err"])
                .groupby("Shape")
                .agg(n=("StoneId", "size"), MAE=("ae", "mean"))
                .round(2).reset_index().sort_values("n", ascending=False))
    return pd.DataFrame([overall]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"}), by_shape


def _market_sheet(ctx: dict, eng: PricingEngine) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = eng.index.as_dict()
    trend = pd.DataFrame(
        [{"Month": m, "Quality-adjusted index (pts)": v} for m, v in idx["monthly_index"].items()]
    )
    trend.loc[len(trend)] = ["recent slope (pts/mo)", idx["recent_slope_pts_per_month"]]
    trend.loc[len(trend)] = ["direction", idx["direction"]]
    macro = pd.DataFrame(ctx["signals"])
    return trend, macro


_LEGEND = [
    ("ACTUAL Disc% / Net$", "REAL — the stone's true realized sale from records.json "
     "(the client's production data). The held-out truth we compare against."),
    ("SUGGESTED Disc% / Net$", "REAL model output — leakage-free quantile GBM trained only on "
     "EARLIER sales, then market-anchored + trend-adjusted. Never saw this stone's outcome."),
    ("Disc Error / Net Error", "COMPUTED — suggestion minus actual. The honest accuracy signal."),
    ("CI Low/High, Actual in band?", "REAL — conformal 80% interval. Note: empirical coverage is "
     "~64% on 6 months of out-of-time data (stated honestly), tightens as data accrues."),
    ("Method", "REAL — 'model+anchor' = model + live market anchor; 'fallback' = sparse-data "
     "hierarchical estimate (rare shape / fancy color), human review."),
    ("Market comps (n) / Market median Disc%", "REAL — from the Uni market dump (268,815 UNIQUE "
     "stones after removing ~90% duplicate re-listings). This is genuine market data."),
    ("BGM state / deduction", "REAL logic. State is 'unassessed' for ALL client stones because "
     "records.json has NO BGM fields yet. Deduction = 0 until BGM is captured. The deduction "
     "VALUES (milky -5/-10/-11, shade -6) are REAL, learned from the Uni market dump."),
    ("Trend shift(pts)", "REAL — damped forward de-bias from the internal price index. 0 here "
     "because these stones are priced as-of their own sale month (no forward gap)."),
    ("Feedback corr(pts)", "REAL — online correction from human overrides. 0 here because no "
     "feedback has been recorded yet (empty decisions log)."),
    ("Market direction", "REAL — computed from the client's own quality-adjusted sales index."),
    ("Why (explanation)", "REAL — deterministic template from the computed numbers. A number "
     "guard rejects any figure not computed. Claude narration activates with ANTHROPIC_API_KEY."),
]

_HONESTY = [
    ("Model training", "COMPLETE & WORKING. Trains in ~30s on load; reproducible. NOTE: the model "
     "is retrained in-memory on startup, not yet saved to disk as a versioned artifact."),
    ("Leakage control", "WORKING. Forbidden/transaction columns physically cannot enter the model "
     "(guard raises). Validation is out-of-time. The accuracy here is honest."),
    ("Market data (Uni)", "REAL but FROM ONE BANKED SNAPSHOT (the 6.2GB dump, ~3-Jun-2026). The "
     "anchor, comparables and BGM deltas are computed from its 268,815 unique stones. Live "
     "re-pulling is NOT happening — no Uni API credentials are set."),
    ("Market RESEARCH (macro: RAPI, lab-grown, tariffs, G7)", "SEEDED / HARDCODED from web research "
     "(provenance-tagged with sources & dates in market/context.py). These are accurate real-world "
     "facts as compiled 2026-06, NOT a live feed. They are surfaced & cross-checked, never fed as "
     "silent model inputs. To be refreshed from source on each cycle."),
    ("Internal trend index", "REAL — computed live from the client's own sales each run."),
    ("Live API ingestion (4 APIs)", "BUILT & TESTED (mocked), NOT RUNNING — credentials in the docx "
     "are compromised and not set. Pipeline falls back to shipped records.json (see terminal.log)."),
    ("Daily snapshot job", "BUILT, NOT SCHEDULED — needs live credentials; one command + cron line."),
    ("Uni request codebook", "PARTIALLY CONFIRMED only (shape/color=D/clarity=IF,VVS1/lab=GIA/"
     "fluor/country). Unconfirmed codes FAIL LOUD by design — must be confirmed before live queries."),
    ("BGM on client stones", "NOT AVAILABLE — records.json has no BGM fields, so every client stone "
     "is 'unassessed' (priced on the No-BGM clean base, flagged). Recommend capturing it in the CRM."),
    ("Feedback loop", "WORKING — but the decisions log is empty (no human decisions recorded yet)."),
    ("Inventory & Gap engines (2 & 3)", "NOT BUILT — out of scope until the Pricing Engine is signed off."),
    ("Nothing is faked", "No random/placeholder numbers anywhere. Every value is computed from real "
     "data or is a sourced macro fact. The only non-live element is the macro research feed (seeded)."),
]


def build(n: int = 100, split_date: str | None = None, out: Path | None = None) -> Path:
    split_date = split_date or SETTINGS.backtest_split_date
    out = out or OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    res, info, ctx, eng = _results_frame(n, split_date)
    summary, by_shape = _accuracy_summary(res)
    trend, macro = _market_sheet(ctx, eng)
    legend = pd.DataFrame(_LEGEND, columns=["Column", "What it is / is it real?"])
    honesty = pd.DataFrame(_HONESTY, columns=["Component", "Status (the honest truth)"])

    meta = pd.DataFrame([
        ["Report", "Glow Star Pricing Engine — 100 held-out stones"],
        ["Trained on", f"sales before {split_date} ({info.n_train:,} stones)"],
        ["Scored (held-out)", f"sales on/after {split_date} ({info.n_test:,}); {len(res)} sampled"],
        ["Engine MAE on this sample", f"{res['Abs Disc Err'].mean():.2f} discount points"],
        ["Macro market view", ctx["overall_direction"]],
        ["Generated from", "records.json + Uni market artifacts (one banked snapshot)"],
    ], columns=["Field", "Value"])

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        meta.to_excel(xl, sheet_name="Overview", index=False)
        res.to_excel(xl, sheet_name="Pricing Results", index=False)
        summary.to_excel(xl, sheet_name="Accuracy Summary", index=False, startrow=0)
        by_shape.to_excel(xl, sheet_name="Accuracy Summary", index=False, startrow=len(summary) + 2)
        trend.to_excel(xl, sheet_name="Market Research", index=False, startrow=0)
        macro.to_excel(xl, sheet_name="Market Research", index=False, startrow=len(trend) + 2)
        legend.to_excel(xl, sheet_name="Legend & Honesty", index=False, startrow=0)
        honesty.to_excel(xl, sheet_name="Legend & Honesty", index=False, startrow=len(legend) + 2)
        _autosize(xl)
    return out


def _autosize(xl) -> None:
    for ws in xl.book.worksheets:
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))


def main() -> None:
    path = build()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
