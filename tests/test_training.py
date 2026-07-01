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
