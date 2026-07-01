"""Price an external stone file the client sends (e.g. a GIA grading export).

Takes a file of stones with NO Rapaport price, looks up Rap deterministically per
stone (reference.rap_lookup), trains the engine on the latest live sold history,
prices each stone against the live Uni market, and writes the clean client report.

Run:  python -m glowstar.reporting.price_file "artifacts/219 GS.xls"
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import ARTIFACTS_DIR
from ..data.history import assemble_sold_history
from ..feedback import store as fbstore
from ..models.engine import PricingEngine, EngineConfig
from ..reference import rap_lookup as RL
from .client_report import _confidence, _fluor_label, _is_broad_market, _note, _why

log = logging.getLogger(__name__)

# GIA "Shape Description" -> the engine's Shape_full vocabulary.
_SHAPE_MAP = {
    "round brilliant cut": "Round", "round brilliant": "Round",
    "oval brilliant": "Oval", "pear brilliant": "Pear",
    "marquise brilliant": "Marquise", "heart brilliant": "Heart",
    "emerald cut": "Emerald", "princess": "Princess",
    "square modified brilliant": "Princess", "cushion brilliant": "Cushion",
    "cushion modified brilliant": "Cushion",
    "rectangular modified brilliant": "Radiant",
    "cut-cornered rectangular modified brilliant": "Radiant",
    "square emerald cut": "Sq. Emerald",
}
_FLUOR_MAP = {"NON": "Non", "FNT": "Fnt", "MED": "Med", "STG": "Stg",
              "VSL": "Vsl", "SLT": "Slt", "VSTG": "Vstg"}


def _shape_full(desc: str) -> str:
    d = str(desc or "").strip().lower()
    return _SHAPE_MAP.get(d, str(desc).split()[0].title() if desc else "NA")


def _make_cps(cut, pol, sym) -> str:
    """CPS code that matches the client's TRAINING vocabulary (3EX / VG / EX / GD / FR).

    CRITICAL: the model learned the cut effect from clean grades. Emitting combined
    codes like 'VG-EX' creates categories it has NEVER seen, so it can't apply the
    cut penalty and under-discounts (a VG stone priced like a 3EX). So we emit
    '3EX' for triple-excellent, else the CUT grade the model knows. For fancy
    shapes (GIA doesn't grade cut) we fall back to polish/symmetry.
    """
    _GRADE = {"EXCELLENT": "EX", "EX": "EX", "IDEAL": "EX", "ID": "EX",
              "VERY GOOD": "VG", "VG": "VG", "GOOD": "GD", "GD": "GD", "G": "GD",
              "FAIR": "FR", "FR": "FR", "POOR": "PR", "PR": "PR"}

    def _g(x):
        x = str(x).strip().upper()
        if x in ("", "NAN", "NONE"):
            return ""
        return _GRADE.get(x, x)
    c, p, s = _g(cut), _g(pol), _g(sym)
    if c:                       # graded cut (rounds): 3EX only if all three excellent
        return "3EX" if (c == "EX" and p == "EX" and s == "EX") else c
    # Fancy shapes have NO overall GIA cut grade. The client codes top make
    # (EX polish + EX symmetry) as '3EX' (2,090 such fancy stones in training),
    # otherwise the LOWER (limiting) of polish/symmetry — matching that vocabulary
    # is what keeps the cut signal alive for fancies instead of an unseen code.
    if p == "EX" and s == "EX":
        return "3EX"
    grades = [g for g in (p, s) if g]
    if not grades:
        return "VG"
    _ORD = {"EX": 0, "VG": 1, "GD": 2, "FR": 3, "PR": 4}
    return max(grades, key=lambda g: _ORD.get(g, 1))   # the worse (lower) grade


def parse_gia_export(path: str | Path) -> pd.DataFrame:
    """Parse a GIA grading export into the engine's input schema, with Rap looked
    up per stone. Adds `Rap_status` so non-exact lookups (gap/oversize) are visible."""
    raw = pd.read_excel(path, engine="xlrd", header=0)
    raw.columns = [str(c).strip() for c in raw.columns]
    rows = []
    for r in raw.to_dict("records"):
        sf = _shape_full(r.get("Shape Description"))
        res = RL.lookup(shape_code=str(r.get("Shape")), shape_full=sf,
                        weight=float(r["Weight"]), color=str(r["Color"]),
                        clarity=str(r["Clarity"]))
        rap = res.price_per_ct if res.ok else res.floor_estimate
        rows.append({
            "StoneId": r.get("Client Ref"), "CertificateNo": r.get("Report No"),
            "Shape_full": sf, "Shape": r.get("Shape"), "Weight": float(r["Weight"]),
            "Color": str(r["Color"]), "Clarity": str(r["Clarity"]),
            "CPS": _make_cps(r.get("Final Cut"), r.get("Polish"), r.get("Symmetry")),
            "Fluorescence": _FLUOR_MAP.get(str(r.get("Fluorescence Intensity")).strip().upper(), "Non"),
            "Lab": "GIA", "Location": "NA",
            "Rap": rap, "Rap_status": res.status.value,
        })
    return pd.DataFrame(rows)


def _feedback_guide() -> pd.DataFrame:
    """The accept/reject/override + reason-code reference for the feedback columns.

    Reason codes are pulled from the system enum so the sheet can never drift from
    what the loader (feedback.intake) actually accepts."""
    from ..feedback.store import ReasonCode
    _DESC = {
        "discount_too_deep": "Price too low / discount too deep",
        "discount_too_shallow": "Price too high / discount too shallow",
        "bgm_present": "Stone has Brown/Green/Milky not captured",
        "make_quality": "Superior/inferior make not reflected",
        "market_moved": "Market shifted since this data",
        "special_situation": "Urgent / memo / special buyer",
        "data_error": "A stone attribute is wrong",
        "rare_item": "Rare shape/size — manual call",
        "other": "Anything else (use 'Your note')",
    }
    rows = [
        ("HOW TO USE", "In each row, fill 'Your decision' with accept, reject, or override. "
         "For reject/override add a Reason code below. For override also fill 'Your price ($/ct)'. "
         "Return the file — we load it so the model learns from your decisions."),
        ("DECISION: accept", "You're happy with the suggestion (confirms it to the model)."),
        ("DECISION: reject", "Wrong, but you're not giving a price — Reason required."),
        ("DECISION: override", "You'd price it differently — Reason + Your price required."),
        ("", ""),
        ("REASON CODE", "When to use it"),
    ]
    rows += [(c.value, _DESC.get(c.value, c.value)) for c in ReasonCode]
    return pd.DataFrame(rows, columns=["Field / code", "Meaning"])


def _ext_row(row: dict, s) -> dict:
    """Clean client row for an external (not-yet-sold) stone — includes the cert,
    the Rap basis, and what the market is asking."""
    base = {
        "Stone ID": row.get("StoneId", ""), "Cert No": row.get("CertificateNo", ""),
        "Shape": row["Shape_full"], "Weight (ct)": row["Weight"],
        "Colour": row["Color"], "Clarity": row["Clarity"], "Cut": row.get("CPS", ""),
        "Fluorescence": _fluor_label(row.get("Fluorescence")), "Lab": row.get("Lab", "GIA"),
        "Rapaport list ($/ct)": round(float(row["Rap"]), 0),
        "Suggested price ($/ct)": s.suggested_ppc,
        "Suggested total ($)": s.suggested_net,
        "% below Rapaport": round(abs(s.suggested_discount), 1),
        "Fair range ($/ct)": f"${float(row['Rap'])*(1+s.ci_discount_low/100):,.0f} – "
                             f"${float(row['Rap'])*(1+s.ci_discount_high/100):,.0f}",
        "Confidence": _confidence(s),
        "Market is asking (% below Rap)": None if s.market_median_discount is None
                                          else round(abs(s.market_median_discount), 1),
        "Market comps (n)": "broad market" if _is_broad_market(s) else s.comparable_count,
        "Note": _note(s),
        "Explanation": _why(row, s),
        # --- Feedback columns: the client fills these and returns the file; the
        # system loader (feedback.intake.ingest_feedback_excel) reads them back
        # into the feedback store so the model learns. Empty by default.
        "Your decision (accept/reject/override)": "",
        "Reason (if reject/override)": "",
        "Your price ($/ct) (if override)": "",
        "Your note": "",
    }
    if row.get("Rap_status") and row["Rap_status"] != "ok":
        base["Note"] = f"Rapaport price is a {row['Rap_status']} estimate — verify. " + base["Note"]
    return base


def price_and_report(path: str | Path, *, live: bool = True, retrain: bool = True,
                     market_led: bool = True, out: Path | None = None,
                     anchor_lambda: float | None = None, prebuilt_tables=None,
                     apply_asking_offset: bool | None = None) -> Path:
    """Price an external stone file and write the client report.

    `anchor_lambda` (optional) overrides the model/market blend weight for this
    run: final = (1-lam)*model + lam*market. `prebuilt_tables` (optional) supplies
    an already-built MarketTables so several runs (e.g. a 75/25 vs 50/50 sample
    pair) can share ONE live-market pull — then the ONLY difference between them
    is the blend weight, not market noise from two separate pulls."""
    out = out or (ARTIFACTS_DIR / (Path(path).stem + "_priced.xlsx"))
    stones = parse_gia_export(path)
    log.info("Parsed %d stones; Rap found for %d (%s).", len(stones),
             stones["Rap"].notna().sum(), dict(stones["Rap_status"].value_counts()))
    stones = stones[stones["Rap"].notna()].reset_index(drop=True)

    eng = None
    if not retrain:
        # Low-memory / fast path: reuse the gated, already live-trained registry
        # model (the CPS fix lives in the PARSER, not the model, so this is fully
        # correct). Avoids re-fitting ~8 GBMs.
        from ..models import registry
        eng, card = registry.load_current()
        if eng is not None:
            log.info("Using gated registry model %s (live-trained).", (card or {}).get("version"))
    if eng is None:
        # Bank a fresh live snapshot, then train on the unioned live sold history.
        if live:
            try:
                from ..pipeline import ingest_records
                ingest_records(prefer_live=True)
            except Exception:
                log.exception("Live pull failed; training on existing banked snapshots.")
        sold = assemble_sold_history()
        eng = PricingEngine(EngineConfig()).fit(sold, feedback_records=fbstore.load_all())
        log.info("Engine trained on %d sold stones.", len(sold))

    # Use the freshly-aggregated (cut-aware) banked market as the base table, and
    # price TO that clean cut+4C market (forward pricing) — what the client does.
    from ..market.anchor import MarketTables
    eng.tables = prebuilt_tables if prebuilt_tables is not None else MarketTables.load()
    eng.cfg.market_led = market_led
    if anchor_lambda is not None:
        eng.cfg.anchor_lambda = float(anchor_lambda)   # override blend weight for this run
    if apply_asking_offset is not None:
        eng.cfg.apply_asking_offset = bool(apply_asking_offset)   # list (False) vs realized (True)
    asof = eng._train_max_date
    stones["MarketSheetDate_dt"] = asof
    stones["OrderDate_dt"] = asof

    # Live Uni market for these stones' segments (else banked / prebuilt artifact).
    if live and prebuilt_tables is None:
        try:
            from ..market.live import LiveMarket
            lm = LiveMarket()
            eng.tables = lm.build_tables(stones, base=eng.tables)
            log.info("Priced against LIVE Uni market (%d segment calls).", lm.calls)
        except Exception:
            log.exception("Live Uni failed; using banked market table.")

    sugg = eng.predict(stones)
    rows = [_ext_row(r, s) for r, s in zip(stones.to_dict("records"), sugg)]
    res = pd.DataFrame(rows)

    summary = pd.DataFrame([
        ["What this is", f"GlowStar AI suggested prices for {len(res)} stones you sent, each with a "
         "plain reason, a fair range and a confidence level."],
        ["Rapaport basis", "Each stone has NO price in your file, so we look up its Rapaport list "
         "price (by shape/size/colour/clarity) and price as a % below it — the trade standard."],
        ["Market check", "‘Market is asking’ shows where comparable stones are listed in the live "
         "market, so you can see our suggestion against it."],
        ["Confidence", "High = many close market matches. Medium = fewer. Low = rare stone / thin "
         "data — please review."],
        ["Brown/Green/Milky", "Not in a GIA report, so each price assumes no BGM (and says so). "
         "Recording milky/shade when you inspect a stone sharpens the price."],
        ["Prices are suggestions", "Computed from market data, never guessed. High-value/rare stones "
         "are flagged for your review."],
    ], columns=["", "Plain-English explanation"])

    glossary = pd.DataFrame([
        ("% below Rapaport", "How far below the Rapaport list price we suggest. The trade's standard quote."),
        ("Rapaport list ($/ct)", "The reference per-carat list price for this exact shape/size/colour/clarity."),
        ("Fluorescence", "How much the stone glows under UV (None → Very Strong). The engine factors it into price."),
        ("Fair range", "The 80% price band we're confident the stone sits in."),
        ("Confidence", "High / Medium / Low — based on how many close market matches we found."),
        ("Market is asking", "Where comparable stones are currently listed in the live market (% below Rap)."),
    ], columns=["Column", "What it means"])

    feedback_guide = _feedback_guide()

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="Summary", index=False)
        res.to_excel(xl, sheet_name="Suggested Prices", index=False)
        glossary.to_excel(xl, sheet_name="What the columns mean", index=False)
        feedback_guide.to_excel(xl, sheet_name="How to give feedback", index=False)
        for ws in xl.book.worksheets:
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(70, max(12, w + 2))
    log.info("Wrote priced report -> %s", out)
    return out


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else str(ARTIFACTS_DIR / "219 GS.xls")
    print("Wrote", price_and_report(path))


if __name__ == "__main__":
    main()
