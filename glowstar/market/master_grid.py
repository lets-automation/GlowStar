"""Look up a stone's cell in the client's Master price grid.

The grid is the client's OWN reference for standard stones. This module maps an
engine stone (shape/weight/colour/clarity/CPS/fluorescence) to its grid cell and
returns the cell's discount + freshness. The engine shows this BESIDE its own
independent suggestion and flags drift — it never copies the grid as its answer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from ..ingestion.master_grid import CURRENT_GRID

log = logging.getLogger(__name__)

# Canonical shape tokens. BOTH sides — the engine's Shape_full AND the grid's own
# shape token — are normalised through `canon_shape`, rather than mapping one onto
# the other. That is load-bearing: verified against the live GetCellsHistory feed
# (all sheets, 130d, 2.29M edits), the client's grid spells oval BOTH ways —
# "F.OVAL" (17,190 cells) and "OVAL" (7,245) — so any one-directional map silently
# misses one of them.
#
# A missed mapping is not cosmetic: the stone then has "no explicit cell" and falls
# through to the interpolated grid-model estimate, which scores MAE 10.4 vs the
# desk's own quote (versus 2.2 for a real cell). That is what manufactured the
# "you're 20 points off your grid" complaint on the marquise/oval stones.
_CANON_SHAPE: dict[str, str] = {
    "ROUND": "ROUND", "RBC": "ROUND", "RB": "ROUND", "BR": "ROUND", "RD": "ROUND",
    "OVAL": "OVAL", "F.OVAL": "OVAL", "F OVAL": "OVAL", "FANCY OVAL": "OVAL",
    "PEAR": "PEAR", "PB": "PEAR",
    "MARQUISE": "MARQUISE", "MB": "MARQUISE",
    "HEART": "HEART", "HB": "HEART",
    "EMERALD": "EMERALD",
    "SQUARE EMERALD": "SQ.EMERALD", "SQ. EMERALD": "SQ.EMERALD",
    "SQ EMERALD": "SQ.EMERALD", "SQEM": "SQ.EMERALD",
    "PRINCESS": "PRINCESS", "CUSHION": "CUSHION", "RADIANT": "RADIANT",
    # THE INVENTORY'S OWN NAMES. The grid carries a RADIANT sheet (16,032 cells)
    # and a CUSHION sheet (14,112), but the client's inventory calls those stones
    # "Cut-Cornered Rectangular", "Cushion Long" and "Cushion Brilliant". With no
    # entry here canon_shape returned None, so 343 live stones could never reach a
    # cell that existed all along — landing in the no-cell bucket, which is the
    # worst in the system (MAE 4.97, band holds 68%).
    #
    # This is the same defect as the model-side shape map, on the other side of
    # the boundary: the cells were there, the spelling wasn't.
    "CUT-CORNERED RECTANGULAR": "RADIANT",
    "CUT CORNERED RECTANGULAR": "RADIANT",
    "CUT-CORNERED RECTANGULAR MODIFIED BRILLIANT": "RADIANT",
    "RECTANGULAR MODIFIED BRILLIANT": "RADIANT",
    "CCRMB": "RADIANT", "CCSMB": "RADIANT", "RMB": "RADIANT",
    "CUSHION LONG": "CUSHION", "CUSHION BRILLIANT": "CUSHION",
    "CUSHION MODIFIED BRILLIANT": "CUSHION", "CMB": "CUSHION", "CB": "CUSHION",
    "DECA BRI": "DECAGONAL",
    "BAGUETTE": "BAGUETTE",
    "DECAGONAL": "DECAGONAL", "DECAGONAL BRILLIANT": "DECAGONAL",
    "CARRE": "CARRE", "CARRE CUT": "CARRE",
}
# Junk shape tokens seen in the live grid feed (mis-populated rows): the `shape`
# field sometimes holds a lab name or a literal "NONE". They are not shapes and
# must never index a cell.
_NOT_A_SHAPE = frozenset({"GIA", "IGI", "HRD", "NONE", "NA", ""})


def canon_shape(s) -> str | None:
    """Canonical shape token for a grid cell OR an engine stone, else None."""
    t = str(s or "").strip().upper()
    if t in _NOT_A_SHAPE:
        return None
    return _CANON_SHAPE.get(t)
# engine fluorescence -> grid token (grid has NON/FNT/MED/STG/VSTG only)
_FLUOR = {
    "NON": "NON", "NONE": "NON", "FNT": "FNT", "FAINT": "FNT",
    "VSL": "FNT", "VERY SLIGHT": "FNT", "SLT": "FNT", "SLIGHT": "FNT",
    "MED": "MED", "MEDIUM": "MED", "STG": "STG", "STRONG": "STG",
    "VSTG": "VSTG", "VERY STRONG": "VSTG",
}

STALE_DAYS = 21          # a cell older than this is flagged (grid moves fast)


@dataclass(frozen=True)
class GridCell:
    discount: float                 # the client's list discount off Rap (their price)
    mfg_discount: float | None      # secondary manufacturing-discount layer
    additional_discount: float | None
    as_of: str                      # date this cell was last edited
    days_old: int | None
    cell_id: str
    size_range: tuple[float, float]

    @property
    def is_stale(self) -> bool:
        return self.days_old is not None and self.days_old > STALE_DAYS

    def as_dict(self) -> dict:
        return {"grid_discount": self.discount, "grid_mfg_discount": self.mfg_discount,
                "grid_additional_discount": self.additional_discount,
                "grid_as_of": self.as_of, "grid_days_old": self.days_old,
                "grid_stale": self.is_stale, "grid_cell": self.cell_id}


def _shape(s) -> str | None:
    return canon_shape(s)


def _fluor(f) -> str:
    return _FLUOR.get(str(f or "").strip().upper(), "NON")


class MasterGrid:
    """The banked Master grid, indexed for O(1)-ish per-stone lookup."""

    def __init__(self, cells: list[dict], as_of: str | None = None):
        self.as_of = as_of
        # index: (shape, color, clarity, cut, fluor) -> sorted list of (lo, hi, cell)
        self._idx: dict[tuple, list] = {}
        for c in cells:
            shapes = c.get("shape") or []
            for sh in shapes:
                canon = canon_shape(sh)      # normalise the GRID side too (F.OVAL -> OVAL)
                if canon is None:
                    continue                 # junk/unmapped shape token: never index it
                key = (canon, c.get("color"), c.get("clarity"),
                       c.get("cut"), c.get("fluorescence"))
                try:
                    lo, hi = float(c["minWeight"]), float(c["maxWeight"])
                except (TypeError, ValueError, KeyError):
                    continue
                self._idx.setdefault(key, []).append((lo, hi, c))
        for v in self._idx.values():
            v.sort(key=lambda t: t[0])

    @classmethod
    def load(cls, path: Path | None = None) -> "MasterGrid | None":
        p = path or CURRENT_GRID
        if not p.exists():
            return None
        payload = json.loads(p.read_text(encoding="utf-8"))
        return cls(payload.get("cells", []), as_of=payload.get("as_of"))

    @property
    def n_cells(self) -> int:
        return sum(len(v) for v in self._idx.values())

    def lookup(self, shape, weight, color, clarity, cps, fluorescence,
               today: date | None = None) -> GridCell | None:
        """Return the grid cell for a stone, or None if the grid doesn't cover it.

        CPS maps directly (grid cut vocab == 3EX/EX/VG/GD/VG-GD/FR). Size is matched
        by the cell's [minWeight, maxWeight] range (the grid's own bracketing)."""
        sh = _shape(shape)
        if sh is None:
            return None
        key = (sh, str(color or "").strip().upper(), str(clarity or "").strip().upper(),
               str(cps or "").strip().upper(), _fluor(fluorescence))
        cells = self._idx.get(key)
        if not cells:
            return None
        w = float(weight)
        for lo, hi, c in cells:
            if lo <= w <= hi:
                return self._to_cell(c, today)
        return None

    @staticmethod
    def _to_cell(c: dict, today: date | None) -> GridCell:
        as_of = str(c.get("createdDate", ""))[:10]
        days_old = None
        try:
            days_old = ((today or datetime.now().date()) - date.fromisoformat(as_of)).days
        except ValueError:
            pass
        return GridCell(
            discount=float(c["discount"]),
            mfg_discount=c.get("mfgDiscount"),
            additional_discount=c.get("additionalDiscount"),
            as_of=as_of, days_old=days_old, cell_id=str(c.get("cellId", "")),
            size_range=(float(c["minWeight"]), float(c["maxWeight"])),
        )
