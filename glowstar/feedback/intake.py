"""Ingest a returned priced workbook into the feedback store.

The Excel deliverable (reporting.price_file) carries fill-in feedback columns.
When the client returns the file with decisions, this loader reads them and
records each into the immutable feedback store (feedback.store) — exactly the
same records the CRM's accept/reject/override control will post once integrated.
The model then learns from them: overrides become gold training labels and build
per-segment corrections; reasons drive analytics (feedback.learning).

This keeps the human-in-the-loop SYSTEM-driven (read -> validate -> record),
never a manual edit to a model or a price.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .store import Decision, ReasonCode, FeedbackRecord, record

log = logging.getLogger(__name__)

# Report column -> meaning. Kept aligned with reporting.price_file._ext_row.
_DECISION_COL = "Your decision (accept/reject/override)"
_REASON_COL = "Reason (if reject/override)"
_OVERRIDE_PPC_COL = "Your price ($/ct) (if override)"
_NOTE_COL = "Your note"

_DECISIONS = {d.value for d in Decision}
_REASONS = {r.value for r in ReasonCode}


def _clean_str(v) -> str:
    """Empty string for blank/NaN/'nan'/'none' cells (Excel blanks read as NaN)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "none", "na") else s


def _norm_decision(v) -> str | None:
    s = _clean_str(v).lower()
    return s if s in _DECISIONS else None


def _norm_reason(v) -> str | None:
    s = _clean_str(v).lower().replace(" ", "_")
    if not s:
        return None                       # blank reason -> None, so validate() catches it
    if s in _REASONS:
        return s
    # tolerate a free-text reason -> OTHER, preserving the text in the note.
    return ReasonCode.OTHER.value


def _to_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def ingest_feedback_excel(path: str | Path, *, user: str = "client",
                          store_path: Path | None = None) -> dict:
    """Read a returned priced workbook and record every filled-in decision.

    Override price is entered as a $/ct; we convert it to a discount off Rap with
    the trade identity  discount = (price_per_ct / Rap) * 100 - 100  (the same
    formula that makes pricing Rap-change-proof). Rows with no decision are
    skipped. Validation errors are collected and reported, never silently dropped.
    """
    df = pd.read_excel(path, sheet_name="Suggested Prices")
    df.columns = [str(c).strip() for c in df.columns]
    if _DECISION_COL not in df.columns:
        raise ValueError(f"No feedback column {_DECISION_COL!r} found — is this a GlowStar priced file?")

    recorded, skipped, errors = 0, 0, []
    for _, r in df.iterrows():
        decision = _norm_decision(r.get(_DECISION_COL))
        if decision is None:
            skipped += 1
            continue
        rap = _to_float(r.get("Rapaport list ($/ct)")) or 0.0
        # new column name "Sale discount ..."; fall back to the old "% below Rapaport"
        # so files sent before the column rename still load.
        sugg_disc = _to_float(r.get("Sale discount (% below Rap)"))
        if sugg_disc is None:
            sugg_disc = _to_float(r.get("% below Rapaport"))
        suggested_discount = -abs(sugg_disc) if sugg_disc is not None else 0.0

        human_discount = None
        note = _clean_str(r.get(_NOTE_COL))
        if decision == Decision.OVERRIDE.value:
            ppc = _to_float(r.get(_OVERRIDE_PPC_COL))
            if ppc is not None and rap > 0:
                human_discount = round(ppc / rap * 100.0 - 100.0, 2)  # $/ct -> discount off Rap

        reason = _norm_reason(r.get(_REASON_COL))
        raw_reason = _clean_str(r.get(_REASON_COL))
        if reason == ReasonCode.OTHER.value and raw_reason:
            note = (f"reason='{raw_reason}'. " + note).strip()

        rec = FeedbackRecord(
            stone_id=str(r.get("Stone ID") or ""), decision=decision,
            suggested_discount=suggested_discount,
            suggested_net=(_to_float(r.get("Sale total ($)"))
                           or _to_float(r.get("Suggested total ($)")) or 0.0),
            shape_full=str(r.get("Shape") or "NA"), weight=_to_float(r.get("Weight (ct)")) or 0.0,
            color=str(r.get("Colour") or "NA"), clarity=str(r.get("Clarity") or "NA"),
            cps=str(r.get("Cut") or "NA"), fluorescence=str(r.get("Fluorescence") or "Non"),
            lab=str(r.get("Lab") or "GIA"), rap=rap,
            reason_code=reason, note=note, human_discount=human_discount, user=user,
        )
        try:
            record(rec, path=store_path)
            recorded += 1
        except ValueError as e:  # missing reason / override price -> report, don't crash
            errors.append(f"{rec.stone_id}: {e}")

    summary = {"recorded": recorded, "skipped_no_decision": skipped, "errors": errors}
    log.info("Feedback intake from %s: %s", path, summary)
    return summary


def _discount(v) -> float | None:
    """Normalise a workbook's ``% below Rap`` value to the engine's sign.

    Client comparison files commonly show their chosen discount as a negative
    number while the generated report shows it as a positive magnitude.  The
    feedback store always uses the engine convention: a discount below Rap is
    negative.
    """
    value = _to_float(v)
    return None if value is None else -abs(value)


def ingest_client_diff_excel(path: str | Path, *, user: str = "client",
                             store_path: Path | None = None,
                             accepted_tolerance: float = 2.0,
                             max_auto_override_variance: float = 15.0) -> dict:
    """Import a client comparison workbook such as ``GS DIFF.xlsx``.

    Unlike a standard returned GlowStar report, this workbook groups stones
    into tabs and supplies the desk's chosen discount in a ``glow price``
    column.  Its named triage tabs are authoritative (``less than`` accepts;
    ``2-`` and ``more than`` become gold overrides).  For an unrecognised tab,
    ``accepted_tolerance`` provides the same numeric fallback. A very large
    difference is not silently learned: it is quarantined because it is usually
    a sold-out stone, a non-price status, or a data-entry error.

    The returned audit summary deliberately includes every quarantine so the
    caller can resolve it before it influences a future retrain.
    """
    if accepted_tolerance < 0 or max_auto_override_variance <= accepted_tolerance:
        raise ValueError("Override variance threshold must be greater than the accepted tolerance.")

    sheets = pd.read_excel(path, sheet_name=None)
    recorded = accepted = overrides = 0
    rows_seen = 0
    quarantined: list[dict] = []
    errors: list[str] = []
    usable_sheet_found = False

    for sheet_name, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        required = {"Stone ID", "Shape", "Weight (ct)", "Colour", "Clarity",
                    "Rapaport list ($/ct)", "Sale discount (% below Rap)", "glow price"}
        if not required.issubset(df.columns):
            continue
        usable_sheet_found = True
        # The client has already triaged the sheets. Preserve that decision:
        # values displayed as exactly 2.0/5.0 can otherwise slip into the wrong
        # bucket because of Excel rounding or a boundary convention mismatch.
        sheet_key = str(sheet_name).strip().casefold()
        if "less than" in sheet_key:
            review_bucket = False
        elif "2-" in sheet_key or "more than" in sheet_key:
            review_bucket = True
        else:
            review_bucket = None

        for _, r in df.iterrows():
            rows_seen += 1
            stone_id = _clean_str(r.get("Stone ID"))
            suggested_discount = _discount(r.get("Sale discount (% below Rap)"))
            client_discount = _discount(r.get("glow price"))
            rap = _to_float(r.get("Rapaport list ($/ct)")) or 0.0
            if not stone_id or suggested_discount is None or client_discount is None or rap <= 0:
                errors.append(f"{sheet_name}: invalid pricing row for {stone_id or 'unknown stone'}")
                continue

            variance = abs(client_discount - suggested_discount)
            if variance > max_auto_override_variance:
                quarantined.append({
                    "stone_id": stone_id,
                    "sheet": sheet_name,
                    "suggested_discount": suggested_discount,
                    "client_discount": client_discount,
                    "variance": round(variance, 2),
                    "reason": "variance exceeds automatic-learning guard",
                })
                continue

            is_within_tolerance = (not review_bucket if review_bucket is not None
                                    else variance <= accepted_tolerance)
            if is_within_tolerance:
                decision = Decision.ACCEPT.value
                reason = None
                accepted += 1
                human_discount = None
            else:
                decision = Decision.OVERRIDE.value
                # A deeper client discount means our price was too high; a
                # shallower one means our discount was too deep.
                reason = (ReasonCode.DISCOUNT_TOO_SHALLOW.value
                          if client_discount < suggested_discount
                          else ReasonCode.DISCOUNT_TOO_DEEP.value)
                overrides += 1
                human_discount = client_discount

            rec = FeedbackRecord(
                stone_id=stone_id, decision=decision,
                suggested_discount=suggested_discount,
                suggested_net=_to_float(r.get("Sale total ($)")) or 0.0,
                shape_full=_clean_str(r.get("Shape")) or "NA",
                weight=_to_float(r.get("Weight (ct)")) or 0.0,
                color=_clean_str(r.get("Colour")) or "NA",
                clarity=_clean_str(r.get("Clarity")) or "NA",
                cps=_clean_str(r.get("Cut")) or "NA",
                fluorescence=_clean_str(r.get("Fluorescence")) or "Non",
                lab=_clean_str(r.get("Lab")) or "GIA", rap=rap,
                reason_code=reason,
                note=(f"Imported from {Path(path).name}; sheet={sheet_name}; "
                      f"client_discount={client_discount:.2f}; variance={variance:.2f}"),
                human_discount=human_discount, user=user,
            )
            try:
                record(rec, path=store_path)
                recorded += 1
            except ValueError as e:
                errors.append(f"{stone_id}: {e}")

    if not usable_sheet_found:
        raise ValueError("No GS comparison sheet found with the required price columns.")

    summary = {
        "rows_seen": rows_seen,
        "recorded": recorded,
        "accepted_within_tolerance": accepted,
        "overrides_for_learning": overrides,
        "quarantined": quarantined,
        "errors": errors,
    }
    log.info("Client comparison intake from %s: %s", path, summary)
    return summary
