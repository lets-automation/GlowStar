"""Tests for the Master-grid lookup and grid-model."""
from __future__ import annotations

import json
from datetime import date

import pytest

from glowstar.market.master_grid import MasterGrid
from glowstar.models.grid_model import GridModel


def _cell(sh, lo, hi, co, cl, cut, fl, disc, dt="2026-07-01"):
    return {"shape": [sh], "minWeight": lo, "maxWeight": hi, "color": co, "clarity": cl,
            "cut": cut, "fluorescence": fl, "discount": disc, "mfgDiscount": -5.5,
            "additionalDiscount": 0.0, "cellId": f"{lo},{hi},{co},{cl},{cut},{fl}",
            "createdDate": f"{dt}T00:00:00"}


def test_lookup_matches_size_range_and_attributes():
    cells = [
        _cell("ROUND", 0.30, 0.39, "F", "VS2", "3EX", "NON", -35.0),
        _cell("ROUND", 0.40, 0.49, "F", "VS2", "3EX", "NON", -41.0),
    ]
    g = MasterGrid(cells)
    # 0.31 falls in the 0.30-0.39 cell; CPS '3EX' maps straight to grid cut.
    c = g.lookup("Round", 0.31, "F", "VS2", "3EX", "Non", today=date(2026, 7, 3))
    assert c is not None and c.discount == -35.0 and c.days_old == 2 and not c.is_stale
    # 0.45 -> the next cell.
    assert g.lookup("Round", 0.45, "F", "VS2", "3EX", "Non").discount == -41.0
    # weight outside any range -> miss.
    assert g.lookup("Round", 0.90, "F", "VS2", "3EX", "Non") is None
    # wrong clarity -> miss.
    assert g.lookup("Round", 0.31, "F", "VS1", "3EX", "Non") is None


def test_fluorescence_and_shape_mapping():
    cells = [_cell("SQUARE EMERALD", 0.50, 0.59, "G", "SI1", "VG", "FNT", -60.0)]
    g = MasterGrid(cells)
    # engine 'Sq. Emerald' + 'Vsl' fluoro map to grid 'SQUARE EMERALD' / 'FNT'.
    assert g.lookup("Sq. Emerald", 0.55, "G", "SI1", "VG", "Vsl").discount == -60.0
    # unmapped shape -> miss (grid doesn't cover it).
    assert g.lookup("Kite", 0.55, "G", "SI1", "VG", "Non") is None


def test_radiant_shape_reachable():
    """Regression: RADIANT is the grid's 3rd-largest shape (~3.2k cells) but was
    absent from the engine->grid shape map, so every radiant stone silently missed
    its real cell and got an interpolated estimate. It must now resolve."""
    cells = [_cell("RADIANT", 0.60, 0.69, "E", "IF", "VG", "NON", -58.0)]
    g = MasterGrid(cells)
    c = g.lookup("Radiant", 0.65, "E", "IF", "VG", "Non")
    assert c is not None and c.discount == -58.0


def test_stale_flag():
    g = MasterGrid([_cell("ROUND", 0.5, 0.59, "D", "IF", "3EX", "NON", -40.0, dt="2026-05-01")])
    c = g.lookup("Round", 0.55, "D", "IF", "3EX", "Non", today=date(2026, 7, 3))
    assert c.days_old == 63 and c.is_stale


def test_load_from_file(tmp_path):
    p = tmp_path / "current.json"
    p.write_text(json.dumps({"as_of": "2026-07-03", "cells": [
        _cell("ROUND", 0.30, 0.39, "F", "VS2", "3EX", "NON", -35.0)]}), encoding="utf-8")
    g = MasterGrid.load(p)
    assert g.n_cells == 1 and g.lookup("Round", 0.35, "F", "VS2", "3EX", "Non").discount == -35.0


def test_grid_model_fits_and_generalises():
    # a synthetic grid (>200 cells): discount deepens with worse colour + bigger size.
    cells = []
    for i, co in enumerate("DEFGHIJK"):
        for j, sz in enumerate([0.35, 0.45, 0.55, 0.75, 0.95, 1.10]):
            for cl in ("VVS2", "VS1", "VS2", "SI1", "SI2"):
                for fl in ("NON", "FNT"):
                    cells.append(_cell("ROUND", sz - 0.04, sz + 0.04, co, cl, "3EX", fl,
                                       -(35 + i * 2 + j * 3)))
    gm = GridModel.fit(cells)
    d = gm.predict("Round", 0.45, "F", "VS2", "3EX", "Non")   # covered region
    assert -60 < d < -30                                       # sensible list discount
    # worse colour prices deeper than better colour (learned structure)
    assert gm.predict("Round", 0.45, "J", "VS2", "3EX", "Non") < gm.predict("Round", 0.45, "D", "VS2", "3EX", "Non")


def test_grid_model_needs_enough_cells():
    with pytest.raises(ValueError):
        GridModel.fit([_cell("ROUND", 0.3, 0.39, "F", "VS2", "3EX", "NON", -35.0)])


def test_grid_check_columns():
    """The grid drift-check emits the right columns for cell / model-estimate /
    off-grid / stale cases (regression: a wiring bug once dropped these)."""
    from types import SimpleNamespace
    from glowstar.reporting.price_file import _grid_check
    from glowstar.market.master_grid import GridCell
    row = {"Rap": 1800.0}
    s = SimpleNamespace(suggested_discount=-45.0, market_median_discount=-45.0)

    # explicit fresh cell, within 2 pts -> "matches your grid"
    cell = GridCell(-44.0, -5.5, 0.0, "2026-07-01", 2, "c", (0.5, 0.59))
    r = _grid_check(row, s, cell)
    assert r["Your Master grid (% below Rap)"] == 44.0 and r["Our vs grid (pts)"] == -1.0
    assert "matches your grid" in r["Grid check"]

    # stale cell where the live market agrees with us, not the grid -> stale flag
    s2 = SimpleNamespace(suggested_discount=-52.0, market_median_discount=-52.0)
    stale = GridCell(-44.0, -5.5, 0.0, "2026-05-01", 63, "c", (0.5, 0.59))
    assert "stale" in _grid_check(row, s2, stale)["Grid check"]

    # Off-grid: a stone with NO explicit cell shows a BLANK grid column — never an
    # interpolated guess, even if one is passed in. This test used to assert the
    # opposite. Scored against the desk's own returned quotes, that estimate was
    # 10.4 MAE (52% of them >=5pts out) versus 2.2 for a real cell, and on small
    # high-colour fancies it printed ~64% where the desk priced ~42% — making our
    # correct price look "20 points off your own grid". That column, not our price,
    # is what the client escalated on.
    r2 = _grid_check(row, s, None, predicted=-50.0)
    assert r2["Your Master grid (% below Rap)"] is None
    assert r2["Our vs grid (pts)"] is None
    assert "interpolat" not in r2["Grid check"].lower()
    assert "not on your grid" in r2["Grid check"]

    # fully off-grid behaves the same way
    r3 = _grid_check(row, s, None, None)
    assert r3["Your Master grid (% below Rap)"] is None and "not on your grid" in r3["Grid check"]


def test_grid_age_and_refresh_if_stale(tmp_path):
    """grid_age_hours reads the bank date; refresh_if_stale skips a fresh bank
    (no API call) and would refresh a stale/missing one."""
    import json
    from datetime import datetime, timedelta
    from glowstar.ingestion.master_grid import grid_age_hours, refresh_if_stale
    p = tmp_path / "current.json"
    # a fresh bank (2h old)
    fresh = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
    p.write_text(json.dumps({"as_of": fresh, "n_cells": 1, "cells": []}), encoding="utf-8")
    age = grid_age_hours(p)
    assert age is not None and 1.5 < age < 3.0
    # fresh -> no refresh, no API call
    assert refresh_if_stale(max_age_hours=24.0, path=p) is False
    # missing bank -> age is None
    assert grid_age_hours(tmp_path / "nope.json") is None


def test_client_round_slots():
    """Round pricing uses the client's exact (irregular) price slots, so 0.84 and
    0.85 land in DIFFERENT slots — not one lumped 0.80-0.89 bucket. Fancies keep
    the 0.10ct bucket."""
    from glowstar.market.segments import round_slot, size_tag, size_bucket_window
    assert round_slot(0.84) == (0.83, 0.84)
    assert round_slot(0.85) == (0.85, 0.89)          # different slot
    assert round_slot(0.32) == (0.32, 0.34)          # irregular low-decade split
    assert round_slot(0.31) == (0.30, 0.31)
    assert round_slot(1.05) is None                  # outside specified range
    # round: distinct tags for 0.84 vs 0.85
    assert size_tag(0.84, "Round") != size_tag(0.85, "Round")
    assert size_bucket_window(0.84, "Round") == (0.83, 0.84)
    # fancy: unchanged 0.10ct bucket (both 0.80-0.89)
    assert size_tag(0.84, "Oval") == size_tag(0.85, "Oval") == "#0.80"
