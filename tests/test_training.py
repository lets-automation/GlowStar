"""Tests for the registry, history assembly, and the promotion gate."""

from __future__ import annotations

import json

import pandas as pd

from glowstar.models import registry
from glowstar.training.retrain import gate_decision


# --- promotion gate (pure logic) ---

def test_gate_promotes_first_model():
    ok, _ = gate_decision(cand_mae=5.0, inc_mae=None)
    assert ok


def test_gate_promotes_when_better_or_within_tolerance():
    assert gate_decision(4.0, 5.0)[0]          # clearly better
    assert gate_decision(5.2, 5.0, tolerance=0.25)[0]  # within wiggle


def test_gate_rejects_materially_worse():
    ok, reason = gate_decision(6.0, 5.0, tolerance=0.25)
    assert not ok and "worse" in reason


def test_gate_rejects_when_no_test_window():
    ok, _ = gate_decision(None, 5.0)
    assert not ok


# --- registry round-trip ---

def test_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    obj = {"weights": [1, 2, 3]}
    card = registry.ModelCard(version="20260619T120000", trained_at="2026-06-19T12:00:00",
                              n_train=100, test_mae=3.88, promoted=True)
    registry.save_engine(obj, card)

    assert registry.current_version() is None        # save does not auto-promote
    registry.set_current(card.version)
    assert registry.current_version() == card.version

    loaded, loaded_card = registry.load_current()
    assert loaded == obj
    assert loaded_card["test_mae"] == 3.88
    assert registry.list_versions() == [card.version]


# --- history assembly: union + dedupe across snapshots ---

def _rec(stone_id: str, fdisc: float, status: str = "Sold") -> dict:
    rap, wt = 8000.0, 1.0
    fnet = rap * (1 + fdisc / 100.0) * wt
    return {
        "StoneId": stone_id, "Shape": "RBC", "Shape_full": "Round", "Weight": wt,
        "Color": "G", "Clarity": "VS2", "Fluorescence": "Non", "CPS": "3EX",
        "Lab": "GIA", "Status": status, "Rap": rap, "Discount": fdisc,
        "FDiscount": fdisc, "NetAmount": fnet, "FNetAmount": fnet,
        "OrderDate": "2026-03-01T00:00:00Z", "MarketSheetDate": "2026-02-01T00:00:00Z",
        "CreatedDate": "2026-01-01T00:00:00Z", "AvailableDays": 10, "Ageing": 10,
    }


def test_assemble_history_unions_and_dedupes(tmp_path, monkeypatch):
    from glowstar.data import history
    monkeypatch.setattr(history, "SNAPSHOT_ROOT", tmp_path)
    snap_dir = tmp_path / "channel_partner"
    snap_dir.mkdir(parents=True)
    # day 1: A, B, C ; day 2: B (updated discount), C, D  -> 4 unique, B from day2
    (snap_dir / "2026-06-01.json").write_text(
        json.dumps([_rec("A", -40), _rec("B", -50), _rec("C", -45)]), encoding="utf-8")
    (snap_dir / "2026-06-02.json").write_text(
        json.dumps([_rec("B", -55), _rec("C", -45), _rec("D", -60)]), encoding="utf-8")

    sold = history.assemble_sold_history()
    assert len(sold) == 4                                   # A, B, C, D
    assert set(sold["StoneId"]) == {"A", "B", "C", "D"}
    b = sold.loc[sold["StoneId"] == "B", "FDiscount"].iloc[0]
    assert b == -55                                          # keep="last" -> day-2 value


# --- the promotion gate must not be a ratchet -------------------------------
# OBSERVED IN PRODUCTION (2026-08-10/11): the first three models went
# 2.469 -> 2.605 -> 2.815 and EVERY promotion was inside tolerance, because the
# gate compared against the incumbent — which is replaced by each promotion, so
# the bar rose with it. Projected forward that reaches MAE ~4.96 in ten nights
# with every log line reading `promoted: True`.

def test_gate_is_not_a_ratchet_replaying_real_production_numbers():
    """Replay the real sequence, then let the same trend continue one more night.

    Under the old incumbent-only rule EVERY night promoted, forever. Under the
    new rule the drift from best-ever accumulates and the gate stops it. The
    two observed nights are still allowed (they are inside the drift budget);
    the third, which is where it would have become a genuine problem, is not.
    """
    from glowstar.training.retrain import gate_decision
    observed = [2.469, 2.605, 2.815]
    best = inc = observed[0]
    for cand in observed[1:]:
        ok, _ = gate_decision(cand, inc, best)
        assert ok, f"{cand} is within budget and should still promote"
        inc, best = cand, min(best, cand)

    # The trend continued at the observed rate (~+0.21/night).
    next_night = 3.025
    ok, why = gate_decision(next_night, inc, best)
    assert not ok, "the ratchet must stop — this is the whole point of the change"
    assert "drifted" in why and str(best) in why


def test_gate_bounds_total_drift_over_many_nights():
    """The whole point: degradation must be bounded, not merely slowed."""
    from glowstar.training.retrain import (gate_decision,
                                           MAX_DRIFT_FROM_BEST_PTS,
                                           PROMOTE_TOLERANCE_PTS)
    best = inc = 2.5
    for _ in range(50):                       # 50 nights of relentless creep
        cand = round(inc + PROMOTE_TOLERANCE_PTS - 0.001, 3)
        ok, _ = gate_decision(cand, inc, best)
        if ok:
            inc = cand
            best = min(best, cand)
    assert inc <= 2.5 + MAX_DRIFT_FROM_BEST_PTS + PROMOTE_TOLERANCE_PTS, \
        f"MAE ratcheted to {inc} — drift is not bounded"


def test_gate_absolute_ceiling_overrides_everything():
    from glowstar.training.retrain import gate_decision, MAX_ACCEPTABLE_MAE
    ok, why = gate_decision(MAX_ACCEPTABLE_MAE + 0.1, inc_mae=99.0, best_mae=99.0)
    assert not ok and "ceiling" in why, "a catastrophic model must never ship"


def test_gate_still_promotes_a_genuine_improvement():
    """Anti-ratchet must not become anti-progress."""
    from glowstar.training.retrain import gate_decision
    ok, _ = gate_decision(2.0, inc_mae=2.6, best_mae=2.4)
    assert ok, "a model better than the best ever must always promote"


def test_best_mae_ignores_corrupt_cards(tmp_path, monkeypatch):
    """A corrupt card must not break the nightly retrain."""
    from glowstar.models import registry
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    (tmp_path / "v1").mkdir(); (tmp_path / "v1" / "metrics.json").write_text('{"test_mae": 3.0}')
    (tmp_path / "v2").mkdir(); (tmp_path / "v2" / "metrics.json").write_text('{"test_mae": 2.1}')
    (tmp_path / "v3").mkdir(); (tmp_path / "v3" / "metrics.json").write_text('not json at all')
    assert registry.best_mae() == 2.1


def test_gate_evaluates_with_point_in_time_grid():
    """The gate must not show a June sale the August grid.

    `predict()` attaches TODAY's grid when the caller supplies none — right for
    serving, wrong for a backtest. `_evaluate` (the gate's own scorer) did not
    attach point-in-time first, so it scored a feature distribution the model was
    never trained on: 2.659 MAE vs 2.408 honest, +0.251 off the truth, on every
    promote/reject decision made so far. CLAUDE.md Trap 5, fourth head.

    Behavioural, not a source grep: an earlier version of this test asserted on
    the text of the function and tripped over a comment. Assert what the code
    DOES — that the grid is joined with no fixed `asof`, i.e. per-row OrderDate.
    """
    from glowstar.training import retrain as R
    from glowstar.training.retrain import serving_config

    seen = {}

    def _spy(df, history, asof=None):
        seen["asof"] = asof
        seen["n"] = len(df)
        out = df.copy()
        out["grid_discount"] = float("nan")
        out["grid_age_days"] = float("nan")
        return out

    class _Eng:
        cfg = serving_config("2026-06-01")
        grid_history = object()          # non-None so the join is attempted
        def predict(self, df, as_of=None):
            return []

    test = pd.DataFrame({
        "FDiscount": [-50.0, -45.0],
        "OrderDate_dt": pd.to_datetime(["2026-06-05", "2026-07-20"]),
    })

    import glowstar.market.grid_history as GH
    orig = GH.attach_grid
    GH.attach_grid = _spy
    try:
        try:
            R._evaluate(_Eng(), test)
        except Exception:
            pass                          # metrics on an empty prediction may fail
    finally:
        GH.attach_grid = orig

    assert "asof" in seen, "_evaluate did not join the grid at all"
    assert seen["asof"] is None, (
        f"_evaluate pinned the grid to {seen['asof']!r}; it must pass no asof so "
        "each row sees the grid as it was on ITS OWN OrderDate")
