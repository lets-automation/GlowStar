"""Tests for Rapaport-list versioning + change detection."""
from __future__ import annotations

from datetime import date

import pytest

from glowstar.reference import rap_versioning as RV


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


# Two list versions for a Round D/IF stone: the 0.90-0.99 cell moves 5000 -> 4500
# (a -10% list cut); the 1.00-1.49 cell is unchanged.
_V1 = [
    ("BR", "IF", "D", 0.90, 0.99, 5000.0, "4/24/2026"),
    ("BR", "IF", "D", 1.00, 1.49, 15000.0, "4/24/2026"),
]
_V2 = [
    ("BR", "IF", "D", 0.90, 0.99, 4500.0, "5/01/2026"),
    ("BR", "IF", "D", 1.00, 1.49, 15000.0, "5/01/2026"),
]


def test_parse_reads_cells_and_date(tmp_path):
    p = tmp_path / "v1.csv"
    _write_csv(p, _V1)
    cells, as_of = RV.parse_rap_csv(p)
    assert as_of == "2026-04-24"
    assert cells[RV.RapCell("round", "D", "IF", 0.90, 0.99)] == 5000.0


def test_ingest_and_list_versions(tmp_path):
    base = tmp_path / "versions"
    _write_csv(tmp_path / "v1.csv", _V1)
    _write_csv(tmp_path / "v2.csv", _V2)
    RV.ingest_list(tmp_path / "v1.csv", base=base)
    RV.ingest_list(tmp_path / "v2.csv", base=base)
    assert RV.list_versions(base) == ["2026-04-24", "2026-05-01"]


def test_diff_finds_only_moved_cells(tmp_path):
    base = tmp_path / "versions"
    _write_csv(tmp_path / "v1.csv", _V1)
    _write_csv(tmp_path / "v2.csv", _V2)
    RV.ingest_list(tmp_path / "v1.csv", base=base)
    RV.ingest_list(tmp_path / "v2.csv", base=base)
    changes = RV.diff(RV.load_version("2026-04-24", base), RV.load_version("2026-05-01", base))
    assert len(changes) == 1
    ch = changes[RV.RapCell("round", "D", "IF", 0.90, 0.99)]
    assert ch.delta_pct == -10.0          # 5000 -> 4500


def test_monitor_flags_stone_in_changed_cell_inside_window(tmp_path):
    base = tmp_path / "versions"
    _write_csv(tmp_path / "v1.csv", _V1)
    _write_csv(tmp_path / "v2.csv", _V2)
    RV.ingest_list(tmp_path / "v1.csv", base=base)
    RV.ingest_list(tmp_path / "v2.csv", base=base)
    mon = RV.RapChangeMonitor(window_days=5, base=base)
    assert mon.n_changed_cells == 1

    # A 0.95ct D/IF round priced 2 days after the move: flagged, in window.
    info = mon.check("Round", 0.95, "D", "IF", pricing_date=date(2026, 5, 3))
    assert info.changed and info.in_window
    assert info.delta_pct == -10.0 and info.days_since == 2

    # The 1.20ct D/IF round (unchanged cell): no flag.
    assert not mon.check("Round", 1.20, "D", "IF", pricing_date=date(2026, 5, 3)).changed

    # Same changed stone priced 10 days later: still "changed" but past the window.
    late = mon.check("Round", 0.95, "D", "IF", pricing_date=date(2026, 5, 13))
    assert late.changed and not late.in_window


def test_monitor_inert_with_single_version(tmp_path):
    base = tmp_path / "versions"
    _write_csv(tmp_path / "v1.csv", _V1)
    RV.ingest_list(tmp_path / "v1.csv", base=base)
    mon = RV.RapChangeMonitor(base=base)
    # Only one version -> nothing to diff -> never guesses a change.
    assert mon.n_changed_cells == 0
    assert not mon.check("Round", 0.95, "D", "IF").changed


def test_fancy_color_never_flagged(tmp_path):
    base = tmp_path / "versions"
    _write_csv(tmp_path / "v1.csv", _V1)
    _write_csv(tmp_path / "v2.csv", _V2)
    RV.ingest_list(tmp_path / "v1.csv", base=base)
    RV.ingest_list(tmp_path / "v2.csv", base=base)
    mon = RV.RapChangeMonitor(base=base)
    # Fancy-colour stones don't price off the white list -> no cell, no flag.
    assert not mon.check("Round", 0.95, "Fancy Yellow", "IF").changed
