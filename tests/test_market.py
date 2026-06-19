"""Tests for the market layer: Uni codebook guard, soft-attribute bucketing,
and the streaming aggregator's classification helpers.
"""

from __future__ import annotations

import pytest

from glowstar.market.aggregate_bulk import _shade_class, _milky_severity, _median_from_hist
from glowstar.market.mappings import to_code, is_confirmed, UnmappedCodeError
from glowstar.market.segments import size_band, segment_keys


# --- Uni codebook: confirmed only, fail loud otherwise (brief Section 3.3) ---

def test_confirmed_codes_resolve():
    # Values verified against the LIVE Uni API (market.calibrate_codebook).
    assert to_code("shape", "Round") == 1
    assert to_code("color", "D") == 1
    assert to_code("clarity", "FL") == 1      # live: 1=FL, 2=IF (doc was wrong)
    assert to_code("clarity", "IF") == 2
    assert to_code("clarity", "VVS1") == 3
    assert to_code("lab", "GIA") == 1
    assert to_code("country", "India") == 99


def test_unknown_value_raises_in_strict_mode():
    # A value with no confirmed/verified code must fail loud, never guess.
    with pytest.raises(UnmappedCodeError):
        to_code("shape", "Kite")          # rare shape, not in the verified codebook
    assert not is_confirmed("shape", "Kite")


# --- Soft-attribute bucketing ---

@pytest.mark.parametrize("raw,exp", [
    ("Faint Brown", "negative"), ("Brown", "negative"), ("Green", "negative"),
    ("Gray", "negative"), ("Mix", "negative"),
    ("None", "none"), ("Not Reported", "none"), ("", "none"),
    ("Pink", "positive"), ("Yellow", "neutral"),
])
def test_shade_classification(raw, exp):
    assert _shade_class(raw) == exp


@pytest.mark.parametrize("raw,exp", [
    ("No Milky", "none"), ("Not Reported", "none"),
    ("Slight Milky", "slight"), ("Medium Milky", "medium"), ("Heavy Milky", "heavy"),
])
def test_milky_severity(raw, exp):
    assert _milky_severity(raw) == exp


# --- Segment helpers ---

def test_size_band_monotonic():
    assert size_band(0.10) < size_band(0.30) < size_band(1.20) < size_band(5.5)


def test_segment_keys_backoff_order():
    keys = segment_keys("Round", 1.2, "G", "VS2")
    assert keys[0] == ("Round", size_band(1.2), "G", "VS2")
    assert keys[-1] == ()                 # global is last


def test_median_from_histogram():
    med, n = _median_from_hist({-50: 1, -52: 1, -54: 1})
    assert med == -52.0 and n == 3
