"""Rapaport-list versioning + change detection (the client's "red line" request).

The Rapaport price list updates ~weekly. When a cell's list price moves, two
things follow that the pricing layer must handle honestly:

  1. **Mechanical:** the new list price flows straight into the price via the
     discount identity (net = Rap*(1+disc/100)*ct) — already correct because the
     engine prices in DISCOUNT space (see models/engine.py).
  2. **Behavioural:** the *market's discount level* re-settles over a few days, so
     a model trained on the old level is briefly less accurate. Every stone whose
     Rap cell just moved must be FLAGGED ("red line"), priced with a wider band,
     and leaned toward the live market until the level re-settles.

This module makes (2) real. It is source-agnostic: feed it each published list
(a CSV today, a RapNet/Diamanto pull when provisioned) and it:
  * stores every version IMMUTABLY and dated (no static single-file dependency),
  * DIFFS consecutive versions to find exactly which cells moved and by how much,
  * exposes a MONITOR that tells the pricing layer, per stone, whether its Rap
    cell changed within the adjustment window (-> flag + widen + lean on market).

No price is ever invented here; it only reports list values and their changes.
"""

from __future__ import annotations

import csv
import json
import logging
from bisect import bisect_right
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path

from ..config import DATA_DIR
from .normalize import RapGrid, clarity_for_grid, grid_for_shape, is_white_grid_color, normalize_color

log = logging.getLogger(__name__)

RAP_VERSIONS_DIR = DATA_DIR / "rap_versions"

# Default adjustment window: how many days after a cell moves we still treat that
# segment as "re-settling" (flag + widen + lean on the live market).
DEFAULT_WINDOW_DAYS = 5

# A cell price move smaller than this (fraction) is treated as noise, not a change.
MIN_REL_MOVE = 0.001


# --- parsing ---------------------------------------------------------------

def _parse_date(s: str | None) -> str | None:
    """Normalise a list date string to ISO (YYYY-MM-DD); None if unparseable."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class RapCell:
    """One published list cell. Coordinates are version-stable; only price moves."""

    grid: str          # RapGrid value: "round" / "fancy"
    color: str
    clarity: str
    lo: float
    hi: float


def parse_rap_csv(path: str | Path) -> tuple[dict[RapCell, float], str | None]:
    """Parse a Rapaport grid CSV into {cell: price} plus the list's as-of date.

    Row format (matches the shipped grids): shape, clarity, color, min, max,
    price[, date]. The shape code maps to the round/fancy grid.
    """
    table: dict[RapCell, float] = {}
    as_of: str | None = None
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 6:
                continue
            shape, clarity, color, lo, hi, price = (c.strip() for c in row[:6])
            if len(row) > 6:
                as_of = as_of or _parse_date(row[6])
            try:
                cell = RapCell(grid_for_shape(shape, None).value, color.upper(),
                               clarity.upper(), float(lo), float(hi))
            except ValueError:
                continue
            table[cell] = float(price)
    return table, as_of


# --- version store ---------------------------------------------------------

def _versions_dir(base: Path | None = None) -> Path:
    return base or RAP_VERSIONS_DIR


def ingest_list(source: str | Path | list, *, as_of: str | None = None,
                base: Path | None = None) -> str:
    """Store a published list as an immutable, dated version. Returns the version id.

    `source` is one CSV or several (a full publish = the Round grid + the Pear/
    fancy grid together); all are merged into one version. `as_of` overrides the
    date parsed from the file(s). Idempotent: re-ingesting the same date overwrites
    that version (lists are re-published, not appended to).
    """
    sources = source if isinstance(source, (list, tuple)) else [source]
    table: dict[RapCell, float] = {}
    file_date: str | None = None
    for src in sources:
        cells, d = parse_rap_csv(src)
        table.update(cells)
        file_date = file_date or d
    version = as_of or file_date or date.today().isoformat()
    d = _versions_dir(base)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": version,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "n_cells": len(table),
        "cells": [[c.grid, c.color, c.clarity, c.lo, c.hi, p] for c, p in table.items()],
    }
    (d / f"{version}.json").write_text(json.dumps(payload), encoding="utf-8")
    log.info("Ingested Rap list %s (%d cells).", version, len(table))
    return version


def list_versions(base: Path | None = None) -> list[str]:
    """All stored version ids (ISO dates), oldest first."""
    d = _versions_dir(base)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_version(version: str, base: Path | None = None) -> dict[RapCell, float]:
    """Load a stored version's {cell: price} map."""
    p = _versions_dir(base) / f"{version}.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {RapCell(g, co, cl, lo, hi): price
            for g, co, cl, lo, hi, price in payload["cells"]}


# --- diff ------------------------------------------------------------------

@dataclass(frozen=True)
class CellChange:
    cell: RapCell
    old: float
    new: float

    @property
    def delta_pct(self) -> float:
        return round((self.new - self.old) / self.old * 100.0, 2) if self.old else 0.0


def diff(old: dict[RapCell, float], new: dict[RapCell, float],
         min_rel_move: float = MIN_REL_MOVE) -> dict[RapCell, CellChange]:
    """Cells whose price moved by more than `min_rel_move` between two versions."""
    out: dict[RapCell, CellChange] = {}
    for cell, new_price in new.items():
        old_price = old.get(cell)
        if old_price is None or old_price == 0:
            continue
        if abs(new_price - old_price) / old_price > min_rel_move:
            out[cell] = CellChange(cell, old_price, new_price)
    return out


# --- monitor ---------------------------------------------------------------

@dataclass(frozen=True)
class RapChangeInfo:
    """What the pricing layer needs to flag a stone after a list move."""

    changed: bool
    delta_pct: float = 0.0
    as_of: str | None = None       # date the new list took effect
    days_since: int | None = None  # days from the list date to the pricing date
    in_window: bool = False        # still inside the adjustment window
    message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


_NO_CHANGE = RapChangeInfo(changed=False)


class RapChangeMonitor:
    """Knows which Rap cells moved in the latest publish and, per stone, whether
    its cell is inside the post-change adjustment window.

    Built from the two most recent stored versions. If fewer than two versions
    exist, it is INERT (every stone reports no change) — never guesses.
    """

    def __init__(self, *, window_days: int = DEFAULT_WINDOW_DAYS, base: Path | None = None):
        self.window_days = window_days
        versions = list_versions(base)
        self.new_version: str | None = versions[-1] if versions else None
        self.prev_version: str | None = versions[-2] if len(versions) >= 2 else None
        if self.prev_version and self.new_version:
            self._changes = diff(load_version(self.prev_version, base),
                                 load_version(self.new_version, base))
            self._new_grid = load_version(self.new_version, base)
        else:
            self._changes, self._new_grid = {}, {}
        # Pre-index changed cells by (grid,color,clarity) -> sorted (lo,hi,change)
        # so a stone's bracket lookup is O(log n).
        self._by_key: dict[tuple, list] = {}
        for cell, ch in self._changes.items():
            self._by_key.setdefault((cell.grid, cell.color, cell.clarity), []).append((cell.lo, cell.hi, ch))
        for v in self._by_key.values():
            v.sort()

    @property
    def n_changed_cells(self) -> int:
        return len(self._changes)

    def _cell_change(self, shape, weight, color, clarity) -> CellChange | None:
        if not is_white_grid_color(str(color)):
            return None
        grid = grid_for_shape(None, str(shape)).value
        co = normalize_color(str(color))
        cl = clarity_for_grid(str(clarity))
        cells = self._by_key.get((grid, co, cl))
        if not cells:
            return None
        i = bisect_right([c[0] for c in cells], float(weight)) - 1
        if i < 0:
            return None
        lo, hi, ch = cells[i]
        return ch if lo <= float(weight) <= hi else None

    def check(self, shape, weight, color, clarity,
              pricing_date: date | None = None) -> RapChangeInfo:
        """Return change info for a stone priced on `pricing_date` (default today)."""
        ch = self._cell_change(shape, weight, color, clarity)
        if ch is None or not self.new_version:
            return _NO_CHANGE
        pricing_date = pricing_date or date.today()
        try:
            days = (pricing_date - date.fromisoformat(self.new_version)).days
        except ValueError:
            days = None
        in_window = days is not None and 0 <= days <= self.window_days
        direction = "up" if ch.delta_pct > 0 else "down"
        msg = (f"Rapaport list for this segment moved {ch.delta_pct:+.1f}% on "
               f"{self.new_version}; the market is still re-settling — wider range, "
               f"leaning on live market." if in_window else
               f"Rapaport list for this segment moved {ch.delta_pct:+.1f}% on {self.new_version}.")
        return RapChangeInfo(changed=True, delta_pct=ch.delta_pct, as_of=self.new_version,
                             days_since=days, in_window=in_window, message=msg)
