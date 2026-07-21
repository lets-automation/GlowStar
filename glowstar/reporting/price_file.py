"""Price an external stone file the client sends (e.g. a GIA grading export).

Takes a file of stones with NO Rapaport price, looks up Rap deterministically per
stone (reference.rap_lookup), trains the engine on the latest live sold history,
prices each stone against the live Uni market, and writes the clean client report.

Run:  python -m glowstar.reporting.price_file "artifacts/219 GS.xls"
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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


# Stone-list shape codes -> engine Shape_full. Covers the client's trade
# abbreviations (RBC=Round, OB=Oval Brilliant, MB=Marquise, PB=Pear, HB=Heart,
# CCRMB=Cut-Cornered Rect. Modified Brilliant=Radiant, SQEM=Square Emerald, ...).
_LIST_SHAPE = {
    "ROUND": "Round", "RBC": "Round", "RB": "Round", "BR": "Round", "RD": "Round",
    "OVAL": "Oval", "OB": "Oval", "OV": "Oval", "S.OV": "Oval", "SOV": "Oval",
    "PEAR": "Pear", "PB": "Pear", "PS": "Pear", "PMB": "Pear",
    "MARQUISE": "Marquise", "MB": "Marquise", "MQ": "Marquise",
    "HEART": "Heart", "HB": "Heart", "HS": "Heart",
    "EMERALD": "Emerald", "EM": "Emerald", "EB": "Emerald", "EC": "Emerald",
    "PRINCESS": "Princess", "PR": "Princess", "SMB": "Princess",
    "CUSHION": "Cushion", "CB": "Cushion", "CU": "Cushion", "CMB": "Cushion",
    "RADIANT": "Radiant", "CCRMB": "Radiant", "RA": "Radiant", "CCSMB": "Radiant",
    "RMB": "Radiant",
    "SQ.EMERALD": "Sq. Emerald", "SQ EMERALD": "Sq. Emerald", "SQEM": "Sq. Emerald",
    "SEM": "Sq. Emerald", "SE": "Sq. Emerald",
}


def _first_col(cols: set, *names: str) -> str | None:
    """First present column name from `names` (case-sensitive after strip)."""
    for n in names:
        if n in cols:
            return n
    return None


def parse_stone_list(path: str | Path) -> pd.DataFrame:
    """Parse a stone-list workbook into the engine's input schema (Rap looked up per
    stone). Robust to column-name variants (Weight/Carats, Fluor/Flour/Fluorescence,
    Stone No/Lot No, Report No/Cert) and, when present, parses the file's OWN
    `BGM Comment` column into milky_ord/brown_ord (the client's assessment for these
    exact stones — authoritative, so it is NOT overwritten by inventory enrichment)."""
    from ..data.loaders import parse_bgm_comments
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = set(raw.columns)
    wcol = _first_col(cols, "Weight", "Carats", "Carat", "Cts", "Cts.")
    icol = _first_col(cols, "Stone No", "Lot No", "StoneId", "Stone ID", "#", "Sr No")
    ccol = _first_col(cols, "Report No", "CertificateNo", "Certificate No", "Cert No", "Certificate")
    fcol = _first_col(cols, "Fluor", "Fluorescence", "Flour", "Fl")
    scol = _first_col(cols, "Sym", "Symmetry")
    bcol = _first_col(cols, "BGM Comment", "BgmComments", "BGM", "BGM Comments")
    if wcol is None or "Color" not in cols or "Clarity" not in cols:
        raise ValueError(f"Unrecognised stone-list layout (need weight+Color+Clarity): {sorted(cols)[:14]}")
    rows = []
    for r in raw.to_dict("records"):
        try:
            wt = float(r[wcol])
        except (TypeError, ValueError, KeyError):
            continue                                  # skip blank/total rows
        if not (0 < wt < 100) or pd.isna(r.get("Color")) or pd.isna(r.get("Clarity")):
            continue
        code = str(r.get("Shape", "")).strip().upper()
        sf = _LIST_SHAPE.get(code, code.title() or "NA")
        res = RL.lookup(shape_code=None, shape_full=sf, weight=wt,
                        color=str(r["Color"]), clarity=str(r["Clarity"]))
        rap = res.price_per_ct if res.ok else res.floor_estimate
        milky_ord, brown_ord = parse_bgm_comments(r.get(bcol)) if bcol else (np.nan, np.nan)
        rows.append({
            "StoneId": str(r.get(icol) or "").strip() if icol else "",
            "CertificateNo": str(r.get(ccol) or "").strip() if ccol else "",
            "Shape_full": sf, "Shape": None, "Weight": wt,
            "Color": str(r["Color"]).strip(), "Clarity": str(r["Clarity"]).strip(),
            "CPS": _make_cps(r.get("Cut"), r.get("Polish"), r.get(scol) if scol else None),
            "Fluorescence": _FLUOR_MAP.get(str(r.get(fcol, "")).strip().upper(), "Non") if fcol else "Non",
            "Lab": str(r.get("LAB") or r.get("Lab") or "GIA").strip().upper(),
            "Location": str(r.get("Location") or "NA"),
            "Rap": rap, "Rap_status": res.status.value,
            "milky_ord": milky_ord, "brown_ord": brown_ord,
        })
    return pd.DataFrame(rows)


_BGM_LVL = {0: "No", 1: "Light", 2: "Medium", 3: "Heavy"}


def _bgm_label(row) -> str:
    """Human BGM status from the enriched milky_ord/brown_ord (from inventory)."""
    def lvl(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(f) else int(round(f))
    m, b = lvl(row.get("milky_ord")), lvl(row.get("brown_ord"))
    if m is None and b is None:
        return "not assessed"
    if (m or 0) == 0 and (b or 0) == 0:
        return "No BGM (clean)"
    return f"{_BGM_LVL.get(b or 0, '?')} Brown, {_BGM_LVL.get(m or 0, '?')} Milky"


def enrich_bgm_from_inventory(stones: pd.DataFrame) -> pd.DataFrame:
    """Attach the client's own BGM (milky_ord/brown_ord) to each stone by matching
    its Stone ID or certificate to the live inventory (records.json `BgmComments`).

    Turns an 'unassessed' external stone into an assessed clean/BGM one, so the
    model prices the milky/brown discount instead of assuming No-BGM. Silent
    no-op (all-unknown) if the inventory or the match is unavailable."""
    from ..data.loaders import parse_bgm_comments
    stones = stones.copy()
    # PRESERVE any BGM already parsed from the input file itself (the client's
    # assessment for these exact stones is authoritative); only fill the rest.
    if "milky_ord" not in stones.columns:
        stones["milky_ord"] = np.nan
    if "brown_ord" not in stones.columns:
        stones["brown_ord"] = np.nan

    # Prefer the LIVE inventory (current stock + BGM); fall back to the banked file.
    inv_records = None
    try:
        from ..ingestion import channel_partner
        inv_records = channel_partner.get_all_records()
    except Exception:
        log.warning("BGM enrichment: live inventory pull failed; trying the banked file.")
        try:
            from ..data.loaders import load_records
            inv_records = load_records()[0].to_dict("records")
        except Exception:
            log.exception("BGM enrichment: no inventory — stones left unassessed.")
            return stones

    id_map: dict[str, tuple] = {}
    cert_map: dict[str, tuple] = {}
    for rec in inv_records:
        m, b = parse_bgm_comments(rec.get("BgmComments"))
        if pd.isna(m) and pd.isna(b):
            continue
        sid = str(rec.get("StoneId") or "")
        cert = str(rec.get("CertificateNo") or "")
        if sid:
            id_map.setdefault(sid, (m, b))
        if cert and cert.lower() not in ("nan", "none"):
            cert_map.setdefault(cert, (m, b))

    hit = 0
    for i, r in stones.iterrows():
        if pd.notna(r.get("milky_ord")) or pd.notna(r.get("brown_ord")):
            continue                              # file already carries BGM — keep it
        found = (id_map.get(str(r.get("StoneId") or ""))
                 or cert_map.get(str(r.get("CertificateNo") or "")))
        if found:
            stones.at[i, "milky_ord"], stones.at[i, "brown_ord"] = found
            hit += 1
    log.info("BGM enrichment: matched %d/%d stones from inventory (rest kept file/NaN).",
             hit, len(stones))
    return stones


def parse_stone_file(path: str | Path) -> pd.DataFrame:
    """Auto-detect the workbook layout (GIA grading export vs stone list)."""
    head = pd.read_excel(path, nrows=1)   # pandas picks the engine by extension
    cols = {str(c).strip() for c in head.columns}
    # A GIA grading export is identified by its distinctive 'Shape Description'
    # column; a stone list carries a weight column + Color + Clarity. (A file can
    # have 'Report No' AND be a stone list — e.g. the client's own 139-stone sheet.)
    if "Shape Description" in cols:
        return parse_gia_export(path)
    if {"Color", "Clarity"} <= cols and ({"Weight", "Carats", "Carat", "Cts"} & cols):
        return parse_stone_list(path)
    raise ValueError(f"Unrecognised stone-file layout: columns {sorted(cols)[:12]}")


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


def _grid_check(row: dict, s, cell, predicted: float | None = None) -> dict:
    """Grid columns: the client's OWN Master-grid cell beside our price, plus a
    drift check. We NEVER copy the grid into our number — we show it for comparison.

    A stone with NO explicit cell shows NO grid number. `predicted` is accepted and
    IGNORED (kept so old callers don't break).

    Why: we used to fill the blank with an interpolated grid-model ESTIMATE, labelled
    "(interpolated)". It was our guess at their sheet, printed in a column headed
    "Your Master grid" — and it was wrong by a mile. Scored against the desk's own
    returned quotes: a REAL cell lands 2.2 MAE, the interpolated estimate 10.4 (52%
    of them >=5pts out). On small high-colour fancies it read ~64% while the desk
    priced them ~42%, so our correct price looked "20 points off your own grid".
    That column, not our price, is what triggered the client escalation. A blank is
    honest; a fabricated reference number is not.
    """
    if cell is None:
        return {"Your Master grid (% below Rap)": None, "Grid cell age (days)": None,
                "Our vs grid (pts)": None,
                "Grid check": "not on your grid — engine-priced (see Sale discount)"}
    our = s.suggested_discount            # negative
    drift = round(our - cell.discount, 1)  # negative = we deeper than grid
    mkt = s.market_median_discount
    if abs(drift) <= 2.0:
        note = "matches your grid (within 2 pts)"
    elif cell.is_stale and mkt is not None and abs(our - mkt) < abs(cell.discount - mkt):
        note = (f"your grid cell is {cell.days_old}d old; live market ≈ "
                f"{abs(mkt):.0f}% vs grid {abs(cell.discount):.0f}% — cell may be stale")
    elif mkt is not None:
        note = (f"differs {drift:+.1f} from your grid; live market ≈ {abs(mkt):.0f}% "
                "— please review")
    else:
        note = f"differs {drift:+.1f} from your grid — please review"
    return {"Your Master grid (% below Rap)": round(abs(cell.discount), 1),
            "Grid cell age (days)": cell.days_old, "Our vs grid (pts)": drift,
            "Grid check": note}


def _ext_row(row: dict, s, cell=None, predicted: float | None = None) -> dict:
    """Clean client row for an external (not-yet-sold) stone — includes the cert,
    the Rap basis, what the market is asking, and the client's own grid + drift."""
    base = {
        "Stone ID": row.get("StoneId", ""), "Cert No": row.get("CertificateNo", ""),
        "Shape": row["Shape_full"], "Weight (ct)": row["Weight"],
        "Colour": row["Color"], "Clarity": row["Clarity"], "Cut": row.get("CPS", ""),
        "Fluorescence": _fluor_label(row.get("Fluorescence")), "Lab": row.get("Lab", "GIA"),
        "BGM (your inventory)": _bgm_label(row),
        "Rapaport list ($/ct)": round(float(row["Rap"]), 0),
        # THREE discounts only (no duplicates):
        #  ASKING  = where comparable stones are LISTED in the live market (reference).
        #  SALE    = OUR suggested realistic price — THE number to use.
        #  (Master grid = the client's own sheet, added by _grid_check below.)
        "Asking discount (% below Rap)": None if s.market_median_discount is None
                                         else round(abs(s.market_median_discount), 1),
        "Sale discount (% below Rap)": round(abs(s.suggested_discount), 1),
        "Sale price ($/ct)": s.suggested_ppc,
        "Sale total ($)": s.suggested_net,
        "Fair range ($/ct)": f"${float(row['Rap'])*(1+s.ci_discount_low/100):,.0f} – "
                             f"${float(row['Rap'])*(1+s.ci_discount_high/100):,.0f}",
        "Confidence": _confidence(s),
        "Market comps (n)": "broad market" if _is_broad_market(s) else s.comparable_count,
        **_grid_check(row, s, cell, predicted),
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
                     market_led: bool = False, out: Path | None = None,
                     anchor_lambda: float | None = None, prebuilt_tables=None,
                     apply_asking_offset: bool | None = None,
                     honor_overrides: bool = True,
                     use_feedback: bool = False) -> Path:
    """Price an external stone file and write the client report.

    `anchor_lambda` (optional) overrides the model/market blend weight for this
    run: final = (1-lam)*model + lam*market. `prebuilt_tables` (optional) supplies
    an already-built MarketTables so several runs (e.g. a 75/25 vs 50/50 sample
    pair) can share ONE live-market pull — then the ONLY difference between them
    is the blend weight, not market noise from two separate pulls."""
    out = out or (ARTIFACTS_DIR / (Path(path).stem + "_priced.xlsx"))
    stones = parse_stone_file(path)
    log.info("Parsed %d stones; Rap found for %d (%s).", len(stones),
             stones["Rap"].notna().sum(), dict(stones["Rap_status"].value_counts()))
    stones = stones[stones["Rap"].notna()].reset_index(drop=True)
    # Attach each stone's own BGM from the client's inventory (milky/brown), so the
    # model prices the milky/brown discount instead of assuming No-BGM.
    stones = enrich_bgm_from_inventory(stones)

    eng = None
    if not retrain:
        # Low-memory / fast path: reuse the gated, already live-trained registry
        # model (the CPS fix lives in the PARSER, not the model, so this is fully
        # correct). Avoids re-fitting ~8 GBMs.
        from ..models import registry
        eng, card = registry.load_current()
        if eng is not None:
            # A registry model is intentionally immutable, but client overrides
            # are an online correction and must take effect before the next
            # scheduled retrain.  Without this, an external price-file run
            # silently ignored newly returned feedback whenever it used the fast
            # path.
            if use_feedback:
                eng.set_feedback(fbstore.load_all())
                if not honor_overrides:
                    # Re-quoting a batch the desk ALREADY priced: replaying their
                    # own override back at them makes the engine look perfect
                    # while measuring nothing. Drop the per-stone echo so the
                    # number in the file is the engine's own.
                    n_echo = len(getattr(eng, "_feedback_overrides", {}) or {})
                    eng._feedback_overrides = {}
                    log.info("Per-stone override echo DISABLED (%d ignored).", n_echo)
            else:
                # Feedback OFF. Measured on the desk's own 122 returned prices:
                # the segment corrections move 96/139 stones by a mean -2.07pts off
                # 3-stone samples and make things WORSE (stones >=5pts off: 12 -> 19;
                # mean err 2.17 -> 2.67). They also cost +0.61 MAE on realized sales.
                # Feedback is still RECORDED for a properly calibrated future use
                # (raise build_corrections min_support ~8-10 and shrink the offsets).
                eng.set_corrections({})
                eng._feedback_overrides = {}
                log.info("Feedback DISABLED for pricing; prices are model+market only.")
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
    # The client's own live Master grid (their reference for standard stones). Shown
    # beside our independent price with a drift check — never copied as our answer.
    # NOTE: no grid-MODEL here. A stone with no explicit cell gets a BLANK grid
    # column, never an interpolated guess at their sheet (see `_grid_check`).
    grid = None
    try:
        from ..market.master_grid import MasterGrid
        from ..ingestion.master_grid import refresh_if_stale
        if live:
            try:
                refresh_if_stale(max_age_hours=24.0)   # never price against a stale grid
            except Exception:
                log.exception("Grid refresh failed; using the banked grid (may be stale).")
        grid = MasterGrid.load()
        if grid is not None:
            log.info("Loaded Master grid: %d cells (as_of %s).", grid.n_cells, grid.as_of)
    except Exception:
        log.exception("Master grid unavailable; report will omit grid columns.")

    def _cell_and_pred(r):
        if grid is None:
            return None, None
        cell = grid.lookup(r["Shape_full"], r["Weight"], r["Color"], r["Clarity"],
                           r.get("CPS"), r.get("Fluorescence"))
        return cell, None

    recs = stones.to_dict("records")
    rows = [_ext_row(r, s, *_cell_and_pred(r)) for r, s in zip(recs, sugg)]
    res = pd.DataFrame(rows)

    summary = pd.DataFrame([
        ["What this is", f"GlowStar AI suggested prices for {len(res)} stones you sent, each with a "
         "plain reason, a fair range and a confidence level."],
        ["Rapaport basis", "Each stone has NO price in your file, so we look up its Rapaport list "
         "price (by shape/size/colour/clarity) and price as a % below it — the trade standard."],
        ["How the price is made", "Our ‘Sale discount’ is computed from TWO inputs only: (1) your own "
         "realized sales history and (2) the live market. Nothing else feeds the number."],
        ["Which discount to use", "USE the ‘Sale discount’ — that is our suggested price. "
         "‘Asking discount’ = where the market is listing (reference). ‘Your Master grid’ = "
         "your own sheet’s number, shown ONLY for side-by-side comparison — it is NOT used to "
         "calculate our price (the price is already final before the grid is even read)."],
        ["Confidence", "High = many close market matches. Medium = fewer. Low = rare stone / thin "
         "data — please review."],
        ["Brown/Green/Milky", "Not in a GIA report, so each price assumes no BGM (and says so). "
         "Recording milky/shade when you inspect a stone sharpens the price."],
        ["Prices are suggestions", "Computed from market data, never guessed. High-value/rare stones "
         "are flagged for your review."],
    ], columns=["", "Plain-English explanation"])

    glossary = pd.DataFrame([
        ("Sale discount (% below Rap)", "★ USE THIS — our suggested price (% below Rap). The realistic level the stone should sell at."),
        ("Sale price ($/ct) / Sale total ($)", "The same Sale suggestion, in $/ct and total $."),
        ("Asking discount (% below Rap)", "Reference only: where comparable stones are currently LISTED in the live market."),
        ("Your Master grid (% below Rap)", "Your OWN price sheet's number for this cell (live), shown for COMPARISON ONLY. Our Sale price is built from your sales history + the live market and does NOT use this grid."),
        ("Grid check", "A side-by-side note on whether our (independently-computed) Sale lands near your grid — a confidence cross-check, not an input to the price."),
        ("Rapaport list ($/ct)", "The reference per-carat list price for this exact shape/size/colour/clarity."),
        ("Fluorescence", "How much the stone glows under UV (None → Very Strong). The engine factors it into price."),
        ("Fair range", "The 80% price band we're confident the stone sits in."),
        ("Confidence", "High / Medium / Low — based on how many close market matches we found."),
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
