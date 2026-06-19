"""Tests for BGM-as-base pricing (client request)."""

from __future__ import annotations

from glowstar.market.bgm import assess


class _StubTables:
    """Minimal MarketTables stand-in exposing soft_delta."""
    def soft_delta(self, milky, shade):
        d = 0.0
        if milky and "Medium" in str(milky):
            d -= 10.0
        if shade and "Brown" in str(shade):
            d -= 6.0
        return d


def test_unassessed_assumes_clean_and_flags():
    a = assess({"Shape_full": "Round", "Color": "G", "Clarity": "SI1"}, _StubTables())
    assert a.state == "unassessed"
    assert a.assumes_no_bgm is True
    assert a.deduction_pts == 0.0
    assert "ASSUMES" in a.note


def test_assessed_clean_no_deduction():
    a = assess({"milky": "No Milky", "Shade": "None"}, _StubTables())
    assert a.state == "clean"
    assert a.deduction_pts == 0.0
    assert a.assumes_no_bgm is False


def test_assessed_bgm_deducts_from_base():
    a = assess({"milky": "Medium Milky", "Shade": "Brown"}, _StubTables())
    assert a.state == "bgm"
    assert a.deduction_pts == -16.0          # -10 milky + -6 shade
    assert a.assumes_no_bgm is False
