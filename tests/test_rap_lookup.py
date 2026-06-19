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


# --- The 6.00-9.99ct gap ---------------------------------------------------

def test_gap_6_to_10_is_explicit_not_silent():
    r = _lk(weight=7.0)
    assert r.status is RapStatus.GAP_6_TO_10
    assert r.price_per_ct is None                # never a silent published value
    assert r.floor_estimate == 100000.0          # 5.00-5.99 cell as labelled floor
    assert "gap" in r.note.lower()


@pytest.mark.parametrize("w", [6.0, 6.5, 9.0, 9.99])
def test_gap_spans_whole_hole(w):
    assert _lk(weight=w).status is RapStatus.GAP_6_TO_10


# --- Oversize and undersize ------------------------------------------------

def test_oversize_above_top_bracket():
    r = _lk(weight=12.0)
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
