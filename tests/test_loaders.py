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


# NOTE: records.json is LIVE data (re-pulled + unioned), so exact counts drift
# every refresh. These tests pin the invariant PROPERTIES/relationships instead
# of frozen counts, so a fresh pull never breaks them (client rule: always live).

def test_counts_match_verified_schema(loaded):
    df, rep = loaded
    assert rep.n_total > 20000                                   # a full book, not a stub
    assert set(rep.status_counts) <= {"Sold", "Stock", "Transit"}
    assert sum(rep.status_counts.values()) == rep.n_total        # partition is exact
    assert rep.status_counts["Sold"] > rep.status_counts["Stock"] > 0
    assert rep.unknown_status_rows == 0


def test_price_identity_holds(loaded):
    """FNetAmount == Rap*(1+FDiscount/100)*Weight for ~all sold stones."""
    _, rep = loaded
    assert rep.identity_checked > 10000
    # essentially exact — a handful of cent-level float mismatches at most
    assert rep.identity_mismatches / rep.identity_checked < 0.001


def test_outliers_flagged_not_dropped(loaded):
    df, rep = loaded
    n_sold = int((df.Status == "Sold").sum())
    assert 0 <= rep.fdiscount_outliers < 0.01 * n_sold           # a tiny fraction (brief 4.3)
    # Outliers remain in the full frame; excluded only from the modeling subset.
    assert len(sold_stones(df, drop_outliers=False)) == n_sold
    assert len(sold_stones(df, drop_outliers=True)) == n_sold - rep.fdiscount_outliers


def test_several_months_of_history(loaded):
    df, _ = loaded
    sold = df[df.Status == "Sold"]
    span_days = (sold["OrderDate_dt"].max() - sold["OrderDate_dt"].min()).days
    assert span_days >= 150                                      # at least ~5-6 months banked


def test_stock_subset(loaded):
    df, _ = loaded
    assert 0 < len(stock_stones(df)) < len(df)                   # a real, non-empty live book


def test_parse_bgm_comments():
    """BgmComments -> (milky_ord, brown_ord). Ordinal & monotone; blank -> NaN."""
    import math
    from glowstar.data.loaders import parse_bgm_comments
    assert parse_bgm_comments("No BROWN NO MILKY") == (0.0, 0.0)
    assert parse_bgm_comments("No BROWN LIGHT MILKY") == (1.0, 0.0)
    assert parse_bgm_comments("MEDIUM BROWN NO MILKY") == (0.0, 2.0)
    assert parse_bgm_comments("No BROWN HEAVY MILKY") == (3.0, 0.0)
    m, b = parse_bgm_comments("")           # unassessed -> NaN
    assert math.isnan(m) and math.isnan(b)


def test_bgm_features_present_and_leakage_free(loaded):
    """milky_ord/brown_ord are built into the model matrix (a real, physical
    inspection feature) and the leakage guard still passes."""
    from glowstar.features.build import build_features
    df, _ = loaded
    x = build_features(df[df["Status"] == "Sold"].head(200))
    assert "milky_ord" in x.columns and "brown_ord" in x.columns
    # no forbidden/transaction column ever leaks in
    from glowstar.data.loaders import FORBIDDEN_FEATURES
    assert not (FORBIDDEN_FEATURES & set(x.columns))
