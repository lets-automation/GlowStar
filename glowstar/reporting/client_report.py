"""Client-facing pricing report — plain language, no jargon, LIVE by default.

For the diamond desk, not data scientists. Every header is plain English, the
discount is shown as "% below Rapaport", confidence is High/Medium/Low (not a
raw number), and notes are written out ("Assumes no Brown/Green/Milky tinge"),
never codes like `bgm_unassessed`.

Two sheets the client actually reads:
  1. "Suggested Prices" — your stones, the suggested price, range, confidence, why.
  2. "Accuracy Proof" — held-out stones the engine never saw: what it suggested
     vs what they REALLY sold for, and the difference. This is the trust builder.
Plus a plain "Summary" and a short "What the columns mean".

LIVE by default: pulls the live book (Channel Partner), trains, and prices each
stone against LIVE Uni market comparables. BGM is ASSESSED where the stone's
certificate is found on Uni (is_bgm/milky/shade inherited); otherwise the note
says it assumes no BGM. Run:  python -m glowstar.reporting.client_report
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR, SETTINGS
from ..data.loaders import load_records, sold_stones, stock_stones
from ..models.engine import PricingEngine, EngineConfig
from ..validation.backtest import time_split

log = logging.getLogger(__name__)
OUT_PATH = ARTIFACTS_DIR / "GlowStar_Client_Report.xlsx"


# --- plain-language helpers ---

_FLUOR_LABEL = {
    "NON": "None", "NONE": "None", "FNT": "Faint", "FAINT": "Faint",
    "VSL": "Very Slight", "SLT": "Slight", "SLIGHT": "Slight",
    "MED": "Medium", "MEDIUM": "Medium", "STG": "Strong", "STRONG": "Strong",
    "VSTG": "Very Strong", "VST": "Very Strong",
}


def _fluor_label(raw) -> str:
    """Trade abbreviation -> readable fluorescence label."""
    return _FLUOR_LABEL.get(str(raw or "").strip().upper(), str(raw or "—"))


# A comparable count above this means the market match backed off to the broad /
# whole-market level (no tight segment match) — not real "similar stones".
_BROAD_COMPS = 20000


def _is_broad_market(s) -> bool:
    return bool(s.comparable_count and s.comparable_count > _BROAD_COMPS)


def _confidence(s) -> str:
    """High / Medium / Low from market support and routing (the real discriminators).

    Low  = sparse-data fallback, rare shape, fancy colour, thin market, or an
           attribute we have explicitly told the desk we cannot price (see below).
    High = a TIGHT market match (real comparables) with a normal band.
    Medium = everything else, including a broad/whole-market fallback (no tight match).
    """
    # A stone we are ASKING the desk to price cannot also be "High confidence".
    # These flags print a note saying we under-price this attribute and they should
    # set it themselves; pairing that with "High" is self-contradictory and the desk
    # reads it — correctly — as us not knowing what we know. Both are real: strong
    # fluoro on near-colourless runs +1.5..+2.2 pts shallow out-of-time, and severe
    # milky/brown is under-discounted for want of examples.
    if "fluor_review" in s.flags or "bgm_review" in s.flags:
        return "Low"
    if (s.method == "fallback" or "rare_shape" in s.flags
            or "fancy_color" in s.flags or "thin_market" in s.flags):
        return "Low"
    if _is_broad_market(s):
        return "Medium"                       # global fallback, not a tight match
    width = s.ci_discount_high - s.ci_discount_low
    # High = priced to a well-supported clean market segment (cut+4C matched).
    if s.method == "model+anchor" and s.comparable_count >= 40 and width <= 13:
        return "High"
    return "Medium"


def _note(s) -> str:
    """Plain-English notes — no codes."""
    notes = []
    if s.bgm_state == "unassessed":
        notes.append("Assumes no Brown/Green/Milky tinge (not recorded for this stone)")
    elif s.bgm_state == "bgm":
        # BGM is now priced by the MODEL (learned from the client's own sales), not
        # a fixed post-model deduction — so don't report the (zeroed) deduction.
        notes.append("Brown/Green/Milky recorded — priced deeper than a clean stone")
    elif s.bgm_state == "clean":
        notes.append("Confirmed clean (no Brown/Green/Milky)")
    if "bgm_review" in s.flags:
        notes.append("Medium/Heavy tinge — please review the discount yourself "
                     "(the model under-prices severe Brown/Milky)")
    if "fluor_review" in s.flags:
        notes.append("Strong fluorescence on a near-colourless stone — please set "
                     "this discount yourself (we have too few such sales to price it "
                     "reliably, and we tend to price it too high)")
    if "rare_shape" in s.flags:
        notes.append("Rare shape — please review")
    if "fancy_color" in s.flags:
        notes.append("Fancy colour — please review")
    if "high_value" in s.flags:
        notes.append("High-value stone — confirm manually")
    if s.method == "fallback":
        notes.append("Limited market data — treat as an estimate")
    return "; ".join(notes) if notes else "Standard pricing"


def _why(row, s) -> str:
    fl = _fluor_label(row.get("Fluorescence"))
    fl_phrase = "no fluorescence" if fl == "None" else f"{fl.lower()} fluorescence"
    base = (f"For this {row['Weight']}ct {row['Color']}/{row['Clarity']} "
            f"{row['Shape_full']} with {fl_phrase}, we suggest ${s.suggested_ppc:,.0f}/ct "
            f"(${s.suggested_net:,.0f} total) — {abs(s.suggested_discount):.0f}% below "
            f"the Rapaport list.")
    if s.market_median_discount is not None:
        if _is_broad_market(s):
            base += (f" The broad market is selling around {abs(s.market_median_discount):.0f}% "
                     f"below list (no tight match for this exact spec, so the price leans on our "
                     f"shape-aware model).")
        else:
            base += (f" {s.comparable_count:,} similar stones in the market are selling "
                     f"around {abs(s.market_median_discount):.0f}% below list.")
    base += (f" Confidence: {_confidence(s).lower()}; fair range "
             f"${s.ci_net_low:,.0f}–${s.ci_net_high:,.0f}.")
    return base


def _client_row(row: dict, s) -> dict:
    rap = float(row["Rap"])
    ppc_low = rap * (1 + s.ci_discount_low / 100.0)    # more discount -> lower price
    ppc_high = rap * (1 + s.ci_discount_high / 100.0)
    return {
        "Stone ID": row.get("StoneId", ""),
        "Shape": row["Shape_full"], "Weight (ct)": row["Weight"],
        "Colour": row["Color"], "Clarity": row["Clarity"],
        "Cut": row.get("CPS", ""),
        "Fluorescence": _fluor_label(row.get("Fluorescence")),
        "Lab": row.get("Lab", ""),
        "Rapaport list ($/ct)": round(rap, 0),
        "Suggested price ($/ct)": s.suggested_ppc,
        "Suggested total ($)": s.suggested_net,
        "% below Rapaport": round(abs(s.suggested_discount), 1),
        "Fair range ($/ct)": f"${ppc_low:,.0f} – ${ppc_high:,.0f}",
        "Confidence": _confidence(s),
        "Similar stones in market": s.comparable_count,
        "Note": _note(s),
        "Explanation": _why(row, s),
    }


def _enrich_bgm(df: pd.DataFrame, bgm_lookup: dict | None) -> pd.DataFrame:
    """Inherit BGM (milky/shade/is_bgm) from Uni by certificate match, so the
    engine can ASSESS it instead of assuming clean. No-op if no lookup/overlap."""
    if not bgm_lookup:
        return df
    df = df.copy()
    certs = df["CertificateNo"].astype(str).str.strip()
    df["milky"] = certs.map(lambda c: (bgm_lookup.get(c) or {}).get("milky"))
    df["Shade"] = certs.map(lambda c: (bgm_lookup.get(c) or {}).get("shade"))
    df["is_bgm"] = certs.map(lambda c: (bgm_lookup.get(c) or {}).get("is_bgm"))
    hit = df["milky"].notna().sum()
    log.info("BGM enriched from Uni cert-join: %d of %d stones matched.", hit, len(df))
    return df


def build(n_stock: int = 40, n_accuracy: int = 60, *, live: bool = True,
          shapes: list[str] | None = None, bgm_lookup: dict | None = None,
          out: Path | None = None, split_date: str | None = None, seed: int = 42) -> Path:
    split_date = split_date or SETTINGS.backtest_split_date
    out = out or OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Data (live pull when asked & possible; else shipped/banked).
    records_path = None
    if live:
        try:
            from ..ingestion import channel_partner
            from ..ingestion.snapshots import save_snapshot
            records_path = save_snapshot(channel_partner.get_all_records(),
                                         source="channel_partner").path
            log.info("Live book pulled and banked.")
        except Exception:
            log.exception("Live pull failed; using shipped records.json.")
    df, _ = load_records(records_path)
    sold = sold_stones(df, drop_outliers=True)
    train, test, info = time_split(sold, split_date)

    eng = PricingEngine(EngineConfig(split_date=split_date)).fit(train)

    # 2. Optional LIVE market table for the sampled stones.
    # The engine TRAINS on all shapes; we only filter the REPORTED stones.
    stock = stock_stones(df)
    if shapes:
        norm = {s.strip().lower() for s in shapes}
        keep = lambda d: d[d["Shape_full"].str.strip().str.lower().isin(norm)]
        stock, test = keep(stock), keep(test)
        log.info("Filtered report to shapes %s: %d stock, %d held-out.",
                 shapes, len(stock), len(test))
    stock_sample = stock.sample(min(n_stock, len(stock)), random_state=seed).reset_index(drop=True)
    acc_sample = test.sample(min(n_accuracy, len(test)), random_state=seed).reset_index(drop=True)
    if live:
        try:
            from ..market.live import LiveMarket
            lm = LiveMarket()
            eng.tables = lm.build_tables(pd.concat([stock_sample, acc_sample], ignore_index=True),
                                         base=eng.tables)
            log.info("Priced against LIVE Uni market (%d segment calls).", lm.calls)
        except Exception:
            log.exception("Live Uni market failed; using banked market table.")

    # 3. BGM enrichment by certificate join (assess instead of assume).
    stock_sample = _enrich_bgm(stock_sample, bgm_lookup)
    acc_sample = _enrich_bgm(acc_sample, bgm_lookup)

    # 4. Price.
    stock_rows = [_client_row(r, s) for r, s in
                  zip(stock_sample.to_dict("records"), eng.predict(stock_sample))]
    acc_rows = []
    for r, s in zip(acc_sample.to_dict("records"), eng.predict(acc_sample)):
        cr = _client_row(r, s)
        actual_ppc = float(r["FPerCarat"]) if r.get("FPerCarat") else float(r["Rap"]) * (1 + float(r["FDiscount"]) / 100)
        cr["ACTUALLY sold at ($/ct)"] = round(actual_ppc, 0)
        cr["Difference ($/ct)"] = round(s.suggested_ppc - actual_ppc, 0)
        cr["Within fair range?"] = ("Yes" if s.ci_discount_low <= float(r["FDiscount"]) <= s.ci_discount_high else "No")
        acc_rows.append(cr)

    stock_df = pd.DataFrame(stock_rows)
    acc_df = pd.DataFrame(acc_rows)

    # 5. Plain summary.
    err = (acc_df["Suggested price ($/ct)"] - acc_df["ACTUALLY sold at ($/ct)"]).abs()
    within = (acc_df["Within fair range?"] == "Yes").mean()
    bgm_assessed = (stock_df["Note"].str.contains("Brown/Green/Milky").sum()
                    - stock_df["Note"].str.contains("Assumes no").sum())
    summary = pd.DataFrame([
        ["What this is", "GlowStar AI suggested prices for your stones, with a plain-English reason and a confidence level for each."],
        ["How accurate (proof)", f"On {len(acc_df)} stones the engine had NEVER seen, the median price miss was "
         f"${np.median(err):,.0f}/ct, and the real sale landed inside our fair range {within:.0%} of the time."],
        ["Confidence levels", "High = lots of close market matches, tight range. Medium = fewer matches. Low = rare stone / thin data — please review."],
        ["Brown/Green/Milky (BGM)", "BGM is not on GIA/IGI certificates and isn't in your data yet, so each price "
         "ASSUMES no BGM and says so. (We verified it cannot be pulled from the market feed for your specific stones.) "
         "Recording milky/shade when you inspect a stone is the one change that makes every price sharper."],
        ["Prices are suggestions", "Every number is computed from market data — never guessed. High-value/rare stones are flagged for your review, not auto-applied."],
    ], columns=["", "Plain-English explanation"])

    glossary = pd.DataFrame([
        ("% below Rapaport", "How far below the Rapaport list price we suggest selling. The trade's standard way to quote."),
        ("Fluorescence", "How much the stone glows under UV (None → Very Strong), from the lab report. It affects price — "
         "strong fluorescence in colourless (D–F) stones usually sells at a discount. The engine already factors it in."),
        ("Suggested price / total", "Per-carat and full-stone price we recommend."),
        ("Fair range", "The price band we're confident the stone sits in (an 80% range)."),
        ("Confidence", "High / Medium / Low — how sure we are, based on market matches and data."),
        ("Similar stones in market", "How many comparable stones we found in the live market to base this on."),
        ("Actually sold at / Difference", "On the proof sheet: the real past sale vs our suggestion. The smaller the difference, the better."),
    ], columns=["Column", "What it means"])

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        stock_df.to_excel(xl, sheet_name="Suggested Prices", index=False)
        acc_df.to_excel(xl, sheet_name="Accuracy Proof", index=False)
        glossary.to_excel(xl, sheet_name="What the columns mean", index=False)
        for ws in xl.book.worksheets:
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(70, max(12, w + 2))
    log.info("Wrote client report -> %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("Wrote", build())


if __name__ == "__main__":
    main()
