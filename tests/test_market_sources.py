"""Tests for the pluggable market-source seam and the macro staleness guard."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from glowstar.market import sources
from glowstar.market.context import current_context


# --- pluggable market sources ---

def test_default_source_is_uni():
    s = sources.get_market_source("uni")
    assert s.name == "uni"
    assert isinstance(s, sources.MarketSource)


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        sources.get_market_source("bogus")


def test_unprovisioned_source_fails_loud_not_silent():
    """RapNet/IDEX must raise with guidance — never return fake/empty market data."""
    for name in ("rapnet", "idex"):
        src = sources.get_market_source(name)
        with pytest.raises(NotImplementedError) as ei:
            src.build_tables(pd.DataFrame())
        assert name in str(ei.value)


# --- macro staleness guard ---

def test_macro_fresh_when_recently_reviewed():
    ctx = current_context(today=dt.date(2026, 6, 15))   # COMPILED_AS_OF = 2026-06
    assert ctx["is_stale"] is False
    assert ctx["compiled_as_of"] == "2026-06"


def test_macro_flagged_stale_when_review_overdue():
    ctx = current_context(today=dt.date(2026, 10, 1))   # ~122 days after 2026-06-01
    assert ctx["is_stale"] is True
    assert ctx["staleness_days"] > 60
    assert "OVERDUE" in ctx["refresh_note"]
