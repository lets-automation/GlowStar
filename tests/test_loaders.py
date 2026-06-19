"""Regression tests for the records loader against the real records.json.

These pin the verified ground-truth facts from the brief (Section 4) so any
future schema drift or loader regression is caught immediately.
"""

from __future__ import annotations

import pytest

from glowstar.data.loaders import load_records, sold_stones, stock_stones


@pytest.fixture(scope="module")
def loaded():
    return load_records()


def test_counts_match_verified_schema(loaded):
    df, rep = loaded
    assert rep.n_total == 28408
    assert rep.status_counts["Sold"] == 20143
    assert rep.status_counts["Stock"] == 8185
    assert rep.status_counts["Transit"] == 80
    assert rep.unknown_status_rows == 0


def test_price_identity_holds(loaded):
    """FNetAmount == Rap*(1+FDiscount/100)*Weight for sold stones (<=2c rounding)."""
    _, rep = loaded
    assert rep.identity_checked == 20143
    assert rep.identity_max_abs_err < 0.02       # pure float rounding, no real mismatch


def test_outliers_flagged_not_dropped(loaded):
    df, rep = loaded
    assert rep.fdiscount_outliers == 19          # brief Section 4.3: 19 of 20,143
    # Outliers remain in the full frame; excluded only from the modeling subset.
    assert len(sold_stones(df, drop_outliers=False)) == 20143
    assert len(sold_stones(df, drop_outliers=True)) == 20143 - 19


def test_six_months_of_history(loaded):
    df, _ = loaded
    sold = df[df.Status == "Sold"]
    assert str(sold["OrderDate_dt"].min().date()) == "2025-12-15"
    assert str(sold["OrderDate_dt"].max().date()) == "2026-06-15"


def test_stock_subset(loaded):
    df, _ = loaded
    assert len(stock_stones(df)) == 8185
