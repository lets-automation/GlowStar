"""Ingest the client's internal Master price grid (Diamanto).

The client's pricing team maintains a live grid — cell = (shape, size-range, colour,
clarity, cut, fluorescence) -> discount off Rap — and updates it constantly
(~240k cell-edits in 14 days). For STANDARD stones, "their price" IS the grid cell.
The engine must therefore read this grid live: show it beside our independent
suggestion, and flag where the live market says a cell has drifted (its real value).

There is no "current full sheet" endpoint, only GetCellsHistory(from,to). We
reconstruct the current grid = the LATEST edit per cell, and MERGE each fresh pull
into a banked, dated store so the grid stays complete and current over time (a
daily job pulls a short window; the first run bootstraps a long one).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from ..config import DATA_DIR
from .diamanto import get_cells_history

log = logging.getLogger(__name__)

GRID_DIR = DATA_DIR / "master_grid"
CURRENT_GRID = GRID_DIR / "current.json"
PRIMARY_SHEET = "Master"          # the client's main reference sheet

# The fields we keep per cell (the grid is huge; keep only what pricing needs).
_KEEP = ("shape", "minWeight", "maxWeight", "color", "clarity", "cut",
         "fluorescence", "discount", "mfgDiscount", "additionalDiscount",
         "oldDiscount", "cellId", "createdDate", "name")


def _slim(r: dict) -> dict:
    return {k: r.get(k) for k in _KEEP}


def _grid_key(shape, cell_id: str) -> str:
    """Unique key for a grid cell.

    `cellId` is 'minW,maxW,color,clarity,cut,fluor' — it does NOT contain the
    shape, so the SAME cellId is reused across shapes/sheets. Keying by cellId
    alone therefore lets one shape's cell silently overwrite another's (e.g. a
    CUSHION cell clobbering the ROUND cell of the same 4C/size). Shape must be
    part of the key.
    """
    return f"{str(shape).upper().strip()}|{cell_id}"


def fetch_grid(days: int = 45, sheet: str | None = None,
               as_of: date | None = None, chunk_days: int = 12) -> dict[str, dict]:
    """Latest edit per (shape, cell) over the last `days`. Returns {key: cell}.

    Pulls in `chunk_days` windows and merges (latest wins): the grid history is
    huge (~250k edits per 2 weeks), so a single long request times out.

    `sheet=None` (the DEFAULT) keeps EVERY sheet. Verified against the live feed:
    the client's grid is spread over many sheets, and the primary "Master" sheet
    holds only a fraction of some shapes — measured over one 3-day window,
    filtering to "Master" dropped 99.8% of CUSHION cells, 81% of PEAR, 61% of
    HEART and 46% of SQUARE EMERALD. Those stones then had no explicit cell and
    were priced by the interpolated estimate (MAE 10.4 vs the desk, versus 2.2
    for a real cell). Pass an explicit sheet name only for diagnostics.
    """
    as_of = as_of or datetime.now().date()
    start = as_of - timedelta(days=days)
    latest: dict[str, dict] = {}
    cur = start
    while cur < as_of:
        end = min(cur + timedelta(days=chunk_days), as_of)
        rows = get_cells_history(cur.isoformat(), end.isoformat())
        for r in rows:
            if sheet is not None and r.get("name") != sheet:
                continue
            cid = r.get("cellId")
            if not cid:
                continue
            d = str(r.get("createdDate", ""))
            for shp in (r.get("shape") or []):
                key = _grid_key(shp, cid)
                prev = latest.get(key)
                if prev is None or d > str(prev.get("createdDate", "")):
                    latest[key] = _slim(r) | {"shape": [shp]}
        log.info("  grid chunk %s..%s: %d cells so far", cur, end, len(latest))
        cur = end
    log.info("Fetched %d grid cells (last %d days, sheet=%s).", len(latest), days,
             sheet or "ALL")
    return latest


def _load_current(path: Path | None = None) -> dict[str, dict]:
    """Banked grid as {(shape|cellId): cell}. Keyed by shape+cellId, never cellId
    alone — see `_grid_key` (one shape's cell would otherwise overwrite another's)."""
    p = path or CURRENT_GRID
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for c in payload.get("cells", []):
        cid = c.get("cellId")
        if not cid:
            continue
        shapes = c.get("shape") or [None]
        for shp in shapes:
            out[_grid_key(shp, cid)] = c if len(shapes) == 1 else (c | {"shape": [shp]})
    return out


def refresh_banked_grid(days: int = 10, bootstrap_days: int = 60,
                        path: Path | None = None) -> dict:
    """Pull recent edits and MERGE them into the banked current grid (latest wins).

    First run (empty store) bootstraps with `bootstrap_days`; later runs pull the
    small `days` window and merge — so the banked grid is always complete + current.
    """
    p = path or CURRENT_GRID
    p.parent.mkdir(parents=True, exist_ok=True)
    banked = _load_current(p)
    window = bootstrap_days if not banked else days
    fresh = fetch_grid(days=window)
    merged, updated = dict(banked), 0
    for key, cell in fresh.items():
        prev = merged.get(key)
        if prev is None or str(cell.get("createdDate", "")) > str(prev.get("createdDate", "")):
            merged[key] = cell
            updated += 1
    payload = {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "n_cells": len(merged), "n_updated": updated, "sheet": "ALL",
        "cells": list(merged.values()),
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    log.info("Banked Master grid: %d cells (%d updated this pull) -> %s", len(merged), updated, p)
    return {"n_cells": len(merged), "n_updated": updated, "window_days": window}


def grid_age_hours(path: Path | None = None) -> float | None:
    """Hours since the banked grid was last refreshed (None if never)."""
    p = path or CURRENT_GRID
    if not p.exists():
        return None
    try:
        as_of = json.loads(p.read_text(encoding="utf-8")).get("as_of")
        return (datetime.now() - datetime.fromisoformat(as_of)).total_seconds() / 3600.0
    except (ValueError, TypeError, KeyError):
        return None


def refresh_if_stale(max_age_hours: float = 24.0, path: Path | None = None) -> bool:
    """Refresh the banked grid if it is missing or older than `max_age_hours`.

    Called before live pricing so the grid column is never silently stale (the
    grid moves ~thousands of cells a day). Best-effort: returns True if it
    refreshed, False if the bank was already fresh; re-raises only on hard failure.
    """
    age = grid_age_hours(path)
    if age is not None and age <= max_age_hours:
        log.info("Master grid is %.1fh old — fresh, no refresh.", age)
        return False
    log.info("Master grid %s — refreshing.", "missing" if age is None else f"{age:.1f}h old")
    refresh_banked_grid(path=path)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(refresh_banked_grid())


if __name__ == "__main__":
    main()
