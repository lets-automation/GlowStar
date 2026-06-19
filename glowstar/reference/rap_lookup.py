"""Deterministic Rapaport price-per-carat lookup (brief Section 7.2).

Given shape, weight, color, clarity this returns the Rapaport $/ct from the
Round/Pear CSV grids. It is 100% deterministic and unit-tested against known
cells. It NEVER returns a silent wrong value: the 6.00-9.99ct gap, oversize
stones, and fancy/cape colors each return an explicit status so the caller can
decide (fallback, human review, or use the per-stone `Rap` already in the data).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

from ..config import PATHS
from .normalize import (
    RapGrid,
    clarity_for_grid,
    grid_for_shape,
    is_white_grid_color,
    normalize_color,
)


class RapStatus(str, Enum):
    """Outcome of a Rap lookup. Anything other than OK needs caller handling."""

    OK = "ok"
    GAP_6_TO_10 = "gap_6_to_10"            # 6.00-9.99ct: no published cell
    OVERSIZE = "oversize"                  # > top published bracket (>10.99ct)
    UNDERSIZE = "undersize"                # < smallest published bracket
    FANCY_OR_CAPE_COLOR = "fancy_or_cape_color"  # not a white D..N color
    NO_CELL = "no_cell"                    # color/clarity combo absent from grid


@dataclass(frozen=True)
class RapResult:
    """Result of a Rap lookup.

    `price_per_ct` is the authoritative published value and is populated ONLY
    when status is OK. For GAP/OVERSIZE we still expose `floor_estimate` (the
    nearest published cell below) so the caller has a documented lower bound,
    clearly labelled as an estimate — never presented as the published price.
    """

    status: RapStatus
    price_per_ct: float | None
    grid: RapGrid | None
    bracket: tuple[float, float] | None
    floor_estimate: float | None
    note: str

    @property
    def ok(self) -> bool:
        return self.status is RapStatus.OK


# Internal grid representation: {(grid, color, clarity): [(min, max, price), ...]}
_GridKey = tuple[RapGrid, str, str]


def _parse_csv(path: Path, grid: RapGrid) -> dict[_GridKey, list[tuple[float, float, float]]]:
    table: dict[_GridKey, list[tuple[float, float, float]]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 6:
                continue
            _shape, clarity, color, min_s, max_s, price = (c.strip() for c in row[:6])
            key = (grid, color.upper(), clarity.upper())
            table.setdefault(key, []).append((float(min_s), float(max_s), float(price)))
    # Keep each cell list sorted by min size for ordered scanning.
    for cells in table.values():
        cells.sort(key=lambda t: t[0])
    return table


@lru_cache(maxsize=1)
def _load_grids() -> dict[_GridKey, list[tuple[float, float, float]]]:
    table = _parse_csv(PATHS.rap_round, RapGrid.ROUND)
    table.update(_parse_csv(PATHS.rap_pear, RapGrid.FANCY))
    return table


def _brackets_for(grid: RapGrid, color: str, clarity: str):
    return _load_grids().get((grid, color, clarity))


def lookup(
    *,
    shape_code: str | None,
    shape_full: str | None,
    weight: float,
    color: str,
    clarity: str,
) -> RapResult:
    """Look up Rapaport $/ct for a stone. Pure and deterministic."""
    grid = grid_for_shape(shape_code, shape_full)

    if not is_white_grid_color(color):
        return RapResult(
            RapStatus.FANCY_OR_CAPE_COLOR, None, grid, None, None,
            f"Color {color!r} is not a white D-N grade; not priceable off the "
            "white Rap list. Route to attribute-based fallback.",
        )

    color_n = normalize_color(color)
    clarity_g = clarity_for_grid(clarity)
    cells = _brackets_for(grid, color_n, clarity_g)
    if not cells:
        return RapResult(
            RapStatus.NO_CELL, None, grid, None, None,
            f"No grid cells for {grid.value}/{color_n}/{clarity_g}.",
        )

    smallest_min = cells[0][0]
    largest_max = cells[-1][1]

    if weight < smallest_min:
        return RapResult(
            RapStatus.UNDERSIZE, None, grid, None, cells[0][2],
            f"Weight {weight} below smallest published bracket "
            f"({smallest_min}). Floor estimate = smallest cell; verify manually.",
        )

    # Exact bracket containing the weight.
    for lo, hi, price in cells:
        if lo <= weight <= hi:
            return RapResult(
                RapStatus.OK, price, grid, (lo, hi), price,
                f"{grid.value} {color_n}/{clarity_g} {lo}-{hi}ct = {price}/ct.",
            )

    # Not in any bracket: either the 6-9.99 gap or oversize. Find the nearest
    # published cell below the weight to expose as a labelled floor estimate.
    below = [(lo, hi, p) for (lo, hi, p) in cells if hi < weight]
    floor = below[-1][2] if below else None

    if weight > largest_max:
        return RapResult(
            RapStatus.OVERSIZE, None, grid, None, floor,
            f"Weight {weight} exceeds top published bracket ({largest_max}). "
            "Oversize stones price by negotiation; floor estimate = top cell.",
        )

    # Inside published range but in an unpublished hole = the 6.00-9.99ct gap.
    return RapResult(
        RapStatus.GAP_6_TO_10, None, grid, None, floor,
        f"Weight {weight} falls in the unpublished 6.00-9.99ct gap. No Rap "
        "cell exists; price by interpolation/negotiation. Floor = 5.99 cell.",
    )
