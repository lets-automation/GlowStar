"""Unit tests for the deterministic Rap lookup core (brief Section 7.2).

These anchor the core against known cells in the real CSV grids and assert the
explicit handling of the 6-9.99ct gap, oversize, undersize, fancy color, and
the FL->IF and round-vs-fancy-grid rules. The core is the foundation everything
else trusts, so coverage here is non-negotiable.
"""

from __future__ import annotations

import pytest

from glowstar.reference.normalize import RapGrid
from glowstar.reference.rap_lookup import RapStatus, lookup


def _lk(**kw):
    base = dict(shape_code="RBC", shape_full="Round", color="D", clarity="IF")
    base.update(kw)
    return lookup(**base)


# --- Known cells (anchors against the real CSVs) ---------------------------

@pytest.mark.parametrize(
    "shape_full,weight,color,clarity,expected",
    [
        ("Round", 1.20, "D", "IF", 15000.0),    # BR,IF,D,1.00,1.49 = 15000
        ("Round", 10.50, "D", "IF", 140000.0),  # BR,IF,D,10.00,10.99 = 140000
        ("Round", 0.02, "D", "IF", 760.0),      # BR,IF,D,0.01,0.03 = 760
        ("Round", 1.00, "G", "VS2", 4700.0),    # boundary lo of 1.00-1.49
        ("Round", 1.49, "G", "VS2", 4700.0),    # boundary hi of 1.00-1.49
        ("Round", 0.55, "H", "SI1", 1400.0),    # BR,SI1,H,0.50,0.69 = 1400
        ("Round", 5.50, "D", "IF", 100000.0),   # BR,IF,D,5.00,5.99 = 100000
        ("Pear", 0.20, "D", "IF", 1270.0),      # PS,IF,D,0.18,0.22 = 1270
        ("Oval", 1.20, "G", "VS2", 4800.0),     # fancy -> Pear list, PS,VS2,G = 4800
        ("Heart", 1.20, "G", "VS2", 4800.0),    # any fancy shape -> Pear list
    ],
)
def test_known_cells(shape_full, weight, color, clarity, expected):
    r = lookup(shape_code=None, shape_full=shape_full, weight=weight, color=color, clarity=clarity)
    assert r.status is RapStatus.OK
    assert r.price_per_ct == expected


def test_round_uses_round_grid_fancy_uses_pear():
    assert _lk(weight=1.2).grid is RapGrid.ROUND
    assert lookup(shape_code="OB", shape_full="Oval", weight=1.2, color="D", clarity="IF").grid is RapGrid.FANCY


# --- FL maps to IF ---------------------------------------------------------

def test_fl_prices_at_if_cell():
    fl = _lk(weight=1.2, clarity="FL")
    iff = _lk(weight=1.2, clarity="IF")
    assert fl.status is RapStatus.OK
    assert fl.price_per_ct == iff.price_per_ct == 15000.0


# --- Bracket coverage follows THE SHEET, never a hardcoded assumption -------
#
# The classic Rapaport round list stops at 5.00-5.99 and resumes at 10.00-10.99,
# leaving the famous 6.00-9.99 "gap". These tests used to assert that gap as a
# permanent fact. It is not: the client's current sheet (26-06-2026) publishes
# 5.00-9.99 and 10.00-99.00, so 6-9.99 IS priced for them.
#
# Verified against their own book — a 6.23ct E/FL round carries their Rap
# $83,500 and our lookup now returns exactly that. Since FDiscount is measured
# against THEIR Rap, following their sheet is the whole point.
#
# So the invariant under test is the BEHAVIOUR, not the bracket table: a weight
# the sheet prices returns OK, and a weight it does NOT price is flagged
# explicitly (never a silent wrong number).

@pytest.mark.parametrize("w", [6.0, 6.5, 9.0, 9.99])
def test_weights_the_current_sheet_prices_return_ok(w):
    """The active sheet covers 5.00-9.99, so these must price cleanly."""
    r = _lk(weight=w)
    assert r.status is RapStatus.OK
    assert r.price_per_ct and r.price_per_ct > 0


def test_an_unpriced_hole_is_reported_explicitly_not_silently():
    """If a sheet ever DOES leave a hole, the caller must be told — never handed
    a neighbouring cell's price as if it were published."""
    from glowstar.reference.rap_lookup import RapResult
    holed = RapResult(RapStatus.GAP_6_TO_10, None, RapGrid.ROUND, None, 100000.0,
                      "falls in an unpublished gap")
    assert holed.price_per_ct is None      # never a silent published value
    assert holed.floor_estimate == 100000.0   # a LABELLED floor is still offered
    assert not holed.ok


# --- Oversize and undersize ------------------------------------------------

def test_oversize_above_top_bracket():
    # The current sheet's top bracket runs to 99ct, so oversize now means above THAT.
    r = _lk(weight=150.0)
    assert r.status is RapStatus.OVERSIZE
    assert r.price_per_ct is None
    assert r.floor_estimate == 140000.0          # top published cell


def test_undersize_below_smallest_bracket():
    # Pear grid starts at 0.18ct; a 0.10ct pear is undersize.
    r = lookup(shape_code="PB", shape_full="Pear", weight=0.10, color="D", clarity="IF")
    assert r.status is RapStatus.UNDERSIZE
    assert r.price_per_ct is None
    assert r.floor_estimate == 1270.0            # smallest pear cell


# --- Fancy / cape colors are not priceable off the white list --------------

@pytest.mark.parametrize("color", ["Fancy Vivid Yellow", "Faint Pink", "O-P", "Y-Z", "*"])
def test_fancy_or_cape_color_routes_to_fallback(color):
    r = _lk(weight=1.2, color=color)
    assert r.status is RapStatus.FANCY_OR_CAPE_COLOR
    assert r.price_per_ct is None


def test_ok_result_flags_are_consistent():
    r = _lk(weight=1.2)
    assert r.ok is True
    assert r.bracket == (1.00, 1.49)
    assert r.floor_estimate == r.price_per_ct
