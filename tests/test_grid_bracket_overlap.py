"""Overlapping grid brackets: the NARROWEST containing bracket must win, in both
the point-in-time history (model feature) and the banked Master grid (client column).

Why this exists (measured 2026-09-02, see GridHistory._read_key docstring): the
client's grid publishes a narrow slot (0.35-0.39) UNDER a coarse overlay
(0.30-0.39) for the same colour/clarity/cut/fluoro, plus foreign-sheet cells that
share the key. "First containing bracket" returned the overlay for 2,127 of
18,075 Round sales - 16.7 pts from where the stone sold, vs 3.2 for the narrow slot.
Through the engine that was -0.25 MAE paired over six weekly origins.

Also guards: the live API must NOT switch desk-feedback corrections on unless
GS_USE_FEEDBACK is set (CLAUDE.md Trap 3).
"""
from __future__ import annotations

from datetime import date

from glowstar.market.grid_history import GridHistory, MATCH_EXACT
from glowstar.market.master_grid import MasterGrid


# --------------------------------------------------------------------------- #
# GridHistory (point-in-time, the model feature)
# --------------------------------------------------------------------------- #

def _hist(**overrides):
    raw = {
        # coarse overlay, edited rarely, deep
        "ROUND|0.3,0.39,F,VS2,3EX,NON": [["2026-06-01T00:00:00", -59.69]],
        # the desk's real slot, edited often, shallow
        "ROUND|0.32,0.33,F,VS2,3EX,NON": [["2026-06-10T00:00:00", -36.0],
                                          ["2026-08-01T00:00:00", -37.0]],
        # a second narrow slot next door (must NOT be chosen for 0.33)
        "ROUND|0.34,0.34,F,VS2,3EX,NON": [["2026-08-01T00:00:00", -30.0]],
    }
    raw.update(overrides)
    return GridHistory(raw)


def test_narrowest_containing_bracket_wins_over_the_overlay():
    h = _hist()
    d, age, lvl = h.as_of_detailed("Round", 0.33, "F", "VS2", "3EX", "Non", "2026-08-14")
    assert d == -37.0, "the 0.32-0.33 slot, not the 0.30-0.39 overlay"
    assert age == 13 and lvl == MATCH_EXACT


def test_overlay_still_answers_where_no_narrow_slot_exists():
    h = _hist()
    d, _, _ = h.as_of_detailed("Round", 0.31, "F", "VS2", "3EX", "Non", "2026-08-14")
    assert d == -59.69


def test_unedited_narrow_slot_falls_back_to_next_narrowest_edited_bracket():
    # Narrow slot exists but its first edit is AFTER the as-of date -> the overlay
    # (edited before) answers. Previously this case returned None outright.
    h = _hist()
    d, _, _ = h.as_of_detailed("Round", 0.33, "F", "VS2", "3EX", "Non", "2026-06-05")
    assert d == -59.69


def test_no_bracket_edited_before_asof_is_none():
    h = _hist()
    assert h.as_of_detailed("Round", 0.33, "F", "VS2", "3EX", "Non", "2026-05-01") == (None, None, None)


def test_point_in_time_is_preserved_inside_the_chosen_bracket():
    h = _hist()
    d, _, _ = h.as_of_detailed("Round", 0.33, "F", "VS2", "3EX", "Non", "2026-07-01")
    assert d == -36.0, "edit of 2026-08-01 must be invisible on 2026-07-01"


def test_equal_width_tie_prefers_lower_lo_deterministically():
    h = GridHistory({
        "ROUND|0.5,0.59,G,VS1,3EX,NON": [["2026-06-01T00:00:00", -50.0]],
        "ROUND|0.55,0.64,G,VS1,3EX,NON": [["2026-06-01T00:00:00", -40.0]],
    })
    d, _, _ = h.as_of_detailed("Round", 0.57, "G", "VS1", "3EX", "Non", "2026-08-01")
    assert d == -50.0


def test_non_overlapping_brackets_unchanged():
    h = GridHistory({
        "PEAR|1.0,1.09,G,VS1,EX,NON": [["2026-06-01T00:00:00", -55.0]],
        "PEAR|1.5,1.59,G,VS1,EX,NON": [["2026-06-01T00:00:00", -50.0]],
    })
    assert h.as_of("Pear", 1.05, "G", "VS1", "EX", "Non", "2026-08-01")[0] == -55.0
    assert h.as_of("Pear", 1.32, "G", "VS1", "EX", "Non", "2026-08-01")[0] is None, "sparse bracket must MISS, not snap"


# --------------------------------------------------------------------------- #
# MasterGrid (the banked grid the client sees in the report)
# --------------------------------------------------------------------------- #

def _cell(lo, hi, disc, dt="2026-07-01", sh="ROUND"):
    return {"shape": [sh], "minWeight": lo, "maxWeight": hi, "color": "F", "clarity": "VS2",
            "cut": "3EX", "fluorescence": "NON", "discount": disc, "mfgDiscount": None,
            "additionalDiscount": 0.0, "cellId": f"{lo},{hi},F,VS2,3EX,NON",
            "createdDate": f"{dt}T00:00:00"}


def test_master_grid_lookup_agrees_with_history_rule():
    g = MasterGrid([_cell(0.30, 0.39, -59.69, "2026-06-01"),
                    _cell(0.32, 0.33, -37.0, "2026-08-01"),
                    _cell(0.34, 0.34, -30.0, "2026-08-01")])
    c = g.lookup("Round", 0.33, "F", "VS2", "3EX", "Non", today=date(2026, 8, 14))
    assert c is not None and c.discount == -37.0 and c.size_range == (0.32, 0.33) and c.days_old == 13
    assert g.lookup("Round", 0.31, "F", "VS2", "3EX", "Non").discount == -59.69
    assert g.lookup("Round", 0.90, "F", "VS2", "3EX", "Non") is None


def test_history_and_master_grid_pick_the_same_cell_for_every_weight():
    """The client column and the model feature must never disagree."""
    cells = [(0.30, 0.39, -59.69), (0.32, 0.33, -37.0), (0.35, 0.39, -41.0), (0.34, 0.34, -30.0)]
    g = MasterGrid([_cell(lo, hi, d) for lo, hi, d in cells])
    h = GridHistory({f"ROUND|{lo},{hi},F,VS2,3EX,NON": [["2026-07-01T00:00:00", d]] for lo, hi, d in cells})
    for w in (0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.37, 0.39):
        gv = g.lookup("Round", w, "F", "VS2", "3EX", "Non")
        hv = h.as_of("Round", w, "F", "VS2", "3EX", "Non", "2026-08-01")[0]
        assert (gv.discount if gv else None) == hv, w


# --------------------------------------------------------------------------- #
# Live feedback corrections stay OFF unless GS_USE_FEEDBACK is set (Trap 3)
# --------------------------------------------------------------------------- #

def test_record_decision_does_not_arm_corrections_when_feedback_is_off(monkeypatch, tmp_path):
    from glowstar.service import pricing_service as ps
    from glowstar.feedback import store as fbstore

    monkeypatch.delenv("GS_USE_FEEDBACK", raising=False)
    monkeypatch.setattr(fbstore, "record", lambda rec, path=None: tmp_path / "d.jsonl")
    monkeypatch.setattr(fbstore, "load_all", lambda path=None: [])

    calls = []

    class _Engine:
        _train_max_date = None

        def set_corrections(self, c):
            calls.append(c)

    svc = ps.PricingService(engine=_Engine(), prefer_registry=False)
    stone = ps.StoneIn(StoneId="T-1", Shape_full="Round", Weight=1.01, Color="G", Clarity="VS1",
                       CPS="3EX", Fluorescence="Non")
    out = svc.record_decision(stone=stone, decision="override", suggested_discount=-45.0,
                              suggested_net=1000.0, human_discount=-48.0, reason_code="market_moved")
    assert out["recorded"] is True
    assert calls == [], "a desk decision must not change the next quote while feedback is off"

    monkeypatch.setenv("GS_USE_FEEDBACK", "1")
    svc.record_decision(stone=stone, decision="override", suggested_discount=-45.0,
                        suggested_net=1000.0, human_discount=-48.0, reason_code="market_moved")
    assert len(calls) == 1, "with the flag set, corrections refresh as before"
