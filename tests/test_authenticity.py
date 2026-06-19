"""Tests for market-data authenticity & cleaning (client priority)."""

from __future__ import annotations

from datetime import datetime

from glowstar.market.authenticity import (
    normalize_uni_stone, clean_market_stones, source_quality, asking_to_transaction,
)


def test_normalize_handles_export_shape():
    raw = {"stone_uni_id": "01-1", "lab": "GIA", "shape": "Round", "size": 0.3,
           "color": "D", "clarity": "IF", "stone_discount": "-54.00%",
           "milky": "No Milky", "shade_name": "None", "is_bgm": "No",
           "certificate_number": "GIA123", "video_url": "http://x"}
    s = normalize_uni_stone(raw)
    assert s["discount"] == -54.0 and s["lab"] == "GIA" and s["has_cert"] and s["has_video"]


def test_normalize_handles_bulk_shape():
    raw = {"diamondID": "9", "lab": {"lab": "IGI"}, "shape": "Oval", "size": 1.0,
           "color": "G", "clarity": "VS1", "price": {"listDiscount": -60},
           "Shade": "Brown", "milky": "Slight Milky", "is_bgm": "Yes"}
    s = normalize_uni_stone(raw)
    assert s["discount"] == -60.0 and s["lab"] == "IGI" and s["shade"] == "Brown"


def test_dedupe_and_report():
    # Same certificate listed 3x (virtual inventory) + one unique.
    dup = {"certificate_number": "C1", "lab": "GIA", "shape": "Round", "size": 1.0,
           "color": "G", "clarity": "VS2", "stone_discount": "-50%"}
    other = {**dup, "certificate_number": "C2", "stone_discount": "-52%"}
    res = clean_market_stones([dup, dict(dup), dict(dup), other])
    assert res.report.n_in == 4
    assert res.report.n_after_dedupe == 2          # C1 collapsed, C2 kept
    assert res.report.duplicate_rate == 0.5
    assert res.report.median_discount is not None


def test_source_quality_orders_labs():
    gia = {"lab": "GIA", "has_cert": True, "has_video": True}
    hrd = {"lab": "HRD", "has_cert": False, "has_video": False}
    assert source_quality(gia) > source_quality(hrd)


def test_asking_to_transaction_applies_offset():
    assert asking_to_transaction(-44.0, -6.0) == -50.0
