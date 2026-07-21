"""Tests for ingesting a returned priced workbook into the feedback store."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from glowstar.feedback.intake import ingest_client_diff_excel, ingest_feedback_excel
from glowstar.feedback.store import Decision, ReasonCode


def _make_workbook(path, decisions):
    """Build a minimal priced workbook with feedback columns filled per `decisions`."""
    rows = []
    for i, (dec, reason, ppc, note) in enumerate(decisions):
        rows.append({
            "Stone ID": f"S{i}", "Shape": "Round", "Weight (ct)": 1.0,
            "Colour": "G", "Clarity": "VS1", "Cut": "VG", "Fluorescence": "Non",
            "Lab": "GIA", "Rapaport list ($/ct)": 10000.0,
            "Suggested price ($/ct)": 5000.0, "Suggested total ($)": 5000.0,
            "% below Rapaport": 50.0,
            "Your decision (accept/reject/override)": dec,
            "Reason (if reject/override)": reason,
            "Your price ($/ct) (if override)": ppc,
            "Your note": note,
        })
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame(rows).to_excel(xl, sheet_name="Suggested Prices", index=False)


def test_ingest_records_accept_reject_override(tmp_path):
    wb = tmp_path / "returned.xlsx"
    store = tmp_path / "decisions.jsonl"
    _make_workbook(wb, [
        ("accept", "", "", ""),                                   # confirm
        ("override", "discount_too_shallow", 4500.0, "too high"), # 4500/10000 -> -55%
        ("reject", "bgm_present", "", "looks milky"),             # reject w/ reason
        ("", "", "", ""),                                          # no decision -> skip
    ])
    summary = ingest_feedback_excel(wb, store_path=store)
    assert summary["recorded"] == 3
    assert summary["skipped_no_decision"] == 1
    assert not summary["errors"]

    recs = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines()]
    by_dec = {r["decision"]: r for r in recs}
    # Override $/ct converted to a discount off Rap via the trade identity.
    assert by_dec[Decision.OVERRIDE.value]["human_discount"] == -55.0
    assert by_dec[Decision.OVERRIDE.value]["reason_code"] == ReasonCode.DISCOUNT_TOO_SHALLOW.value
    assert by_dec[Decision.REJECT.value]["reason_code"] == ReasonCode.BGM_PRESENT.value


def test_reject_without_reason_is_reported_not_crashed(tmp_path):
    wb = tmp_path / "returned.xlsx"
    store = tmp_path / "decisions.jsonl"
    _make_workbook(wb, [("reject", "", "", "no reason given")])
    summary = ingest_feedback_excel(wb, store_path=store)
    assert summary["recorded"] == 0
    assert len(summary["errors"]) == 1          # surfaced, not silently dropped


def test_free_text_reason_falls_back_to_other(tmp_path):
    wb = tmp_path / "returned.xlsx"
    store = tmp_path / "decisions.jsonl"
    _make_workbook(wb, [("reject", "buyer haggled", "", "")])
    summary = ingest_feedback_excel(wb, store_path=store)
    assert summary["recorded"] == 1
    rec = json.loads(store.read_text(encoding="utf-8").splitlines()[0])
    assert rec["reason_code"] == ReasonCode.OTHER.value
    assert "buyer haggled" in rec["note"]       # free text preserved


def test_rejects_non_glowstar_file(tmp_path):
    wb = tmp_path / "other.xlsx"
    with pd.ExcelWriter(wb, engine="openpyxl") as xl:
        pd.DataFrame([{"foo": 1}]).to_excel(xl, sheet_name="Suggested Prices", index=False)
    with pytest.raises(ValueError):
        ingest_feedback_excel(wb)


def test_ingest_client_diff_turns_variances_into_audited_feedback(tmp_path):
    wb = tmp_path / "GS DIFF.xlsx"
    store = tmp_path / "decisions.jsonl"
    accepted_row = [
        # Within two points: confirms the existing suggestion.
        {"Stone ID": "A", "Shape": "Round", "Weight (ct)": 1.0, "Colour": "G",
         "Clarity": "VS1", "Cut": "EX", "Fluorescence": "None", "Lab": "GIA",
         "Rapaport list ($/ct)": 10000, "Sale discount (% below Rap)": 50,
         "glow price": -51.5, "Sale total ($)": 5000},
    ]
    override_row = [
        # The review tab is authoritative even at an exact two-point boundary.
        {"Stone ID": "B", "Shape": "Round", "Weight (ct)": 1.0, "Colour": "G",
         "Clarity": "VS1", "Cut": "EX", "Fluorescence": "None", "Lab": "GIA",
         "Rapaport list ($/ct)": 10000, "Sale discount (% below Rap)": 50,
         "glow price": -52, "Sale total ($)": 5000},
    ]
    quarantined_row = [
        # A sold-out/status-like value must not poison learning.
        {"Stone ID": "C", "Shape": "Round", "Weight (ct)": 1.0, "Colour": "G",
         "Clarity": "VS1", "Cut": "EX", "Fluorescence": "None", "Lab": "GIA",
         "Rapaport list ($/ct)": 10000, "Sale discount (% below Rap)": 50,
         "glow price": -7, "Sale total ($)": 5000},
    ]
    with pd.ExcelWriter(wb, engine="openpyxl") as xl:
        pd.DataFrame(accepted_row).to_excel(xl, sheet_name="less than 2", index=False)
        pd.DataFrame(override_row).to_excel(xl, sheet_name="2-4.99", index=False)
        pd.DataFrame(quarantined_row).to_excel(xl, sheet_name="more than 5", index=False)

    summary = ingest_client_diff_excel(wb, store_path=store)
    assert summary["rows_seen"] == 3
    assert summary["recorded"] == 2
    assert summary["accepted_within_tolerance"] == 1
    assert summary["overrides_for_learning"] == 1
    assert summary["quarantined"][0]["stone_id"] == "C"

    recs = [json.loads(l) for l in store.read_text(encoding="utf-8").splitlines()]
    override = next(r for r in recs if r["stone_id"] == "B")
    assert override["human_discount"] == -52.0
    assert override["reason_code"] == ReasonCode.DISCOUNT_TOO_SHALLOW.value
