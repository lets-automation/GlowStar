"""Test the Excel report generator produces a well-formed, honest report."""

from __future__ import annotations

import openpyxl

from glowstar.reporting.excel_report import build


def test_report_builds_with_expected_sheets(tmp_path):
    out = build(n=15, out=tmp_path / "report.xlsx")
    wb = openpyxl.load_workbook(out)
    assert set(wb.sheetnames) == {
        "Overview", "Pricing Results", "Accuracy Summary", "Market Research", "Legend & Honesty"
    }
    results = wb["Pricing Results"]
    assert results.max_row == 16          # 15 stones + header
    header = [c.value for c in results[1]]
    # The comparison columns the client asked for must be present.
    for col in ("ACTUAL Disc%", "SUGGESTED Disc%", "Disc Error(pts)",
                "Market median Disc%", "Why (explanation)", "BGM state"):
        assert col in header
