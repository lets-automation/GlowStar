"""Spread premium bands, Average Market Make, and the variance threshold.

These pin the client's 2026-07 rules exactly as the desk stated them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glowstar.market.spread import (PREMIUM_BANDS, band_for_weight, diameter,
                                    is_premium_spread, annotate)
from glowstar.market.market_make import (_mean_grade, COLOR_SCALE, CLARITY_SCALE,
                                         average_spec, average_market_make, _cps_token)
from glowstar.feedback.store import (FeedbackRecord, Decision, ReasonCode,
                                     VARIANCE_REASON_THRESHOLD_PTS, needs_attention,
                                     variance_pts)


# --------------------------------------------------------------------------
# Spread / diameter premium
# --------------------------------------------------------------------------
def test_diameter_is_the_mean_of_length_and_width():
    """A round is measured 4.55 x 4.59; the trade quotes the average."""
    assert diameter(4.55, 4.59) == pytest.approx(4.57)
    assert np.isnan(diameter(None, 4.5))
    assert np.isnan(diameter(0, 0))


def test_the_clients_exact_example():
    """Desk: a 0.35-0.39 stone UNDER 4.50 is not premium; 4.55-4.59 qualifies and
    uses the 4.50-4.59 band."""
    # under the band -> not premium
    assert is_premium_spread("Round", 0.37, 4.45, 4.49) is False
    # inside the band (measures 4.55 x 4.59 -> diameter 4.57) -> premium
    assert is_premium_spread("Round", 0.37, 4.55, 4.59) is True
    assert band_for_weight(0.37).label == "4.50-4.59"


def test_band_edges_are_inclusive():
    """The desk's own example counts 4.50 as inside the 4.50-4.59 band."""
    assert is_premium_spread("Round", 0.37, 4.50, 4.50) is True
    assert is_premium_spread("Round", 0.37, 4.59, 4.59) is True
    assert is_premium_spread("Round", 0.37, 4.60, 4.60) is False


def test_every_band_from_the_clients_table():
    for wlo, whi, dlo, dhi in PREMIUM_BANDS:
        w = (wlo + whi) / 2
        assert is_premium_spread("Round", w, dlo, dlo) is True
        assert is_premium_spread("Round", w, dhi, dhi) is True
        assert is_premium_spread("Round", w, dlo - 0.01, dlo - 0.01) is False


def test_uncovered_stone_is_unknown_NOT_a_failure():
    """None != False. A 1.20ct round has no band in the table, so it must not be
    silently excluded from an average it belongs in."""
    assert is_premium_spread("Round", 1.20, 6.8, 6.8) is None   # weight not in table
    assert is_premium_spread("Oval", 0.37, 4.55, 4.59) is None  # diameter is a ROUND concept
    assert is_premium_spread("Round", 0.37, None, None) is None # unmeasured


def test_annotate_survives_a_file_with_no_measurements():
    """External GIA files often carry no Length/Width — they must still price."""
    df = pd.DataFrame([{"Shape_full": "Round", "Weight": 0.37,
                        "Color": "E", "Clarity": "VS1"}])
    out = annotate(df)
    assert out["diameter"].isna().all()
    assert out["is_premium_spread"].iloc[0] is None
    assert out["spread_band"].iloc[0] == "4.50-4.59"   # band still known from weight


# --------------------------------------------------------------------------
# Average Market Make
# --------------------------------------------------------------------------
def test_average_of_a_grade_is_by_rank():
    """'Average colour of D E F' = E, not a number."""
    assert _mean_grade(["D", "E", "F"], COLOR_SCALE) == "E"
    assert _mean_grade(["D", "D", "F"], COLOR_SCALE) == "E"     # mean rank 0.67 -> E
    assert _mean_grade(["IF", "VVS1", "VVS2"], CLARITY_SCALE) == "VVS1"
    assert _mean_grade([], COLOR_SCALE) is None
    assert _mean_grade(["ZZ"], COLOR_SCALE) is None             # unknown grade ignored


def test_cps_collapses_to_the_leading_cut_grade():
    assert _cps_token("EX-EX-EX") == "3EX"
    assert _cps_token("3EX") == "3EX"
    assert _cps_token("VG-GD") == "VG"


def test_average_spec_matches_the_clients_worked_row():
    """Their table row: D E F / IF VVS1 VVS2 / EX-EX-EX / NON FNT MED
    -> averages to E / VVS1 / 3EX / Faint."""
    g = pd.DataFrame({
        "Color": ["D", "E", "F"], "Clarity": ["IF", "VVS1", "VVS2"],
        "CPS": ["EX-EX-EX"] * 3, "Fluorescence": ["Non", "Fnt", "Med"],
        "diameter": [4.52, 4.55, 4.58],
    })
    spec = average_spec(g)
    assert spec["avg_color"] == "E"
    assert spec["avg_clarity"] == "VVS1"
    assert spec["avg_cps"] == "3EX"
    assert spec["avg_fluor"] == "Faint"
    assert spec["avg_diameter"] == pytest.approx(4.55)


def test_market_is_matched_AFTER_averaging_and_on_all_criteria():
    """The desk was explicit: select market data only once the averages exist, and
    match on ALL criteria together. A market stone failing ANY criterion is out."""
    stones = pd.DataFrame({
        "Shape_full": ["Round"] * 3, "Weight": [0.36, 0.37, 0.38],
        "Color": ["D", "E", "F"], "Clarity": ["IF", "VVS1", "VVS2"],
        "CPS": ["EX-EX-EX"] * 3, "Fluorescence": ["Non", "Fnt", "Med"],
        "Length": [4.52, 4.55, 4.58], "Width": [4.52, 4.55, 4.58],
    })
    market = pd.DataFrame([
        # matches the averaged spec E/VVS1/3EX/Faint inside the 4.50-4.59 band
        *[{"color": "E", "clarity": "VVS1", "cut": "3EX", "fluorescence": "F",
           "diameter": 4.55, "discount": -40.0} for _ in range(6)],
        # right spec but OUTSIDE the diameter band -> excluded
        {"color": "E", "clarity": "VVS1", "cut": "3EX", "fluorescence": "F",
         "diameter": 4.20, "discount": -90.0},
        # wrong clarity -> excluded even though everything else matches
        {"color": "E", "clarity": "SI1", "cut": "3EX", "fluorescence": "F",
         "diameter": 4.55, "discount": -90.0},
    ])
    (g,) = average_market_make(stones, market)
    assert g.size_group == "0.35-0.39"
    assert g.avg_color == "E" and g.avg_clarity == "VVS1"
    assert g.n_market == 6, "only the fully-matching market stones may count"
    assert g.market_make == pytest.approx(-40.0), "the -90 outliers must be excluded"


def test_thin_market_reports_no_number_rather_than_a_bad_one():
    stones = pd.DataFrame({
        "Shape_full": ["Round"], "Weight": [0.37], "Color": ["E"], "Clarity": ["VVS1"],
        "CPS": ["3EX"], "Fluorescence": ["Non"], "Length": [4.55], "Width": [4.55],
    })
    market = pd.DataFrame([{"color": "E", "clarity": "VVS1", "cut": "3EX",
                            "fluorescence": "N", "diameter": 4.55, "discount": -40.0}])
    (g,) = average_market_make(stones, market)
    assert g.market_make is None and "too few" in g.note


# --------------------------------------------------------------------------
# Variance threshold — reason optional below it, required above
# --------------------------------------------------------------------------
def test_variance_and_attention():
    assert variance_pts(-50.0, -52.0) == pytest.approx(2.0)
    assert variance_pts(-50.0, None) is None
    assert not needs_attention(-50.0, -51.0)          # 1 pt: ordinary judgement
    assert not needs_attention(-50.0, -52.0)          # exactly 2: still inside
    assert needs_attention(-50.0, -52.5)              # past the threshold
    assert needs_attention(-50.0, -55.0)              # 5 pts: needs a look


def _rec(**kw):
    base = dict(stone_id="X", decision=Decision.OVERRIDE.value, suggested_discount=-50.0,
                suggested_net=1000.0, shape_full="Round", weight=1.0, color="G",
                clarity="VS1")
    base.update(kw)
    return FeedbackRecord(**base)


def test_small_override_needs_NO_reason():
    """Client rule: a small gap is normal trading judgement. Forcing a code there
    just trains the desk to pick junk, which poisons the reason analytics."""
    _rec(human_discount=-51.0).validate()             # 1 pt, no reason -> fine
    _rec(human_discount=-52.0).validate()             # exactly at the threshold


def test_large_override_DOES_need_a_reason():
    with pytest.raises(ValueError, match="reason_code is required"):
        _rec(human_discount=-58.0).validate()         # 8 pts, no reason
    # with a reason it is accepted
    _rec(human_discount=-58.0, reason_code=ReasonCode.MARKET_MOVED.value).validate()


def test_override_still_requires_a_price_and_reject_still_requires_a_reason():
    """Relaxing the reason must not relax what makes a record USELESS."""
    with pytest.raises(ValueError, match="corrected discount"):
        _rec(human_discount=None).validate()
    with pytest.raises(ValueError, match="reason_code"):
        _rec(decision=Decision.REJECT.value, human_discount=None, reason_code=None).validate()


def test_record_exposes_variance_and_attention_to_the_caller():
    r = _rec(human_discount=-57.0)
    assert r.variance == pytest.approx(7.0)
    assert r.needs_attention is True
    assert _rec(human_discount=-51.0).needs_attention is False
    assert VARIANCE_REASON_THRESHOLD_PTS == 2.0       # client-set 2026-07
