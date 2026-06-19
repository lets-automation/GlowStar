"""Tests for the market-trend index and macro-research context."""

from __future__ import annotations

import pandas as pd

from glowstar.data.loaders import load_records, sold_stones
from glowstar.market.index import MarketIndex
from glowstar.market.context import current_context, cross_check


def test_index_builds_monthly_series_and_drift():
    df, _ = load_records()
    sold = sold_stones(df)
    idx = MarketIndex().fit(sold)
    assert idx.series is not None and len(idx.series) >= 6
    # Drift is additive and self-consistent.
    a, b = idx.series.index[0], idx.series.index[-1]
    assert abs(idx.drift(a, b) - (idx.level(b) - idx.level(a))) < 1e-9


def test_index_projects_beyond_last_month():
    df, _ = load_records()
    idx = MarketIndex().fit(sold_stones(df))
    future = idx.last_period + 2
    # Projection moves from the last level by ~2 * recent slope.
    expected = idx.series.iloc[-1] + idx.recent_slope * 2
    assert abs(idx.project(future) - expected) < 1e-6


def test_macro_context_has_provenance():
    ctx = current_context()
    assert ctx["overall_direction"] in {"softening", "firming", "flat"}
    assert all(s["source"] for s in ctx["signals"])      # every signal is sourced
    assert all(s["as_of"] for s in ctx["signals"])


def test_cross_check_flags_divergence():
    assert cross_check("softening")["agree"] is True      # macro is softening
    assert cross_check("firming")["agree"] is False        # disagreement surfaced
