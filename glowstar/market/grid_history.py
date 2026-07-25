"""Point-in-time store of the client's Master grid.

WHY THIS EXISTS
---------------
The single biggest driver of our error is not the stone — it is the CURRENT LEVEL
of the stone's price cell. Measured on held-out sales: correcting the per-cell
level (oracle) halves MAE (2.64 -> 1.36) and cuts the >=5pt tail 13.1% -> 3.7%,
while correcting a global or per-week level does almost nothing. The model cannot
supply that level itself — a tree cannot extrapolate time, and 40% of served
stones fall beyond its training window, where it silently returns the last month
it ever saw.

The client's Master grid IS a live per-cell level, maintained by their own desk
(~240k cell edits a fortnight). So it is exactly the missing signal.

WHY POINT-IN-TIME
-----------------
Reading TODAY's grid to explain a PAST sale is leakage — the cell may have been
edited after the stone sold. That contamination is not academic: with the live
grid, a grid/market blend appeared to beat the engine (MAE 1.73 vs 2.46); with a
correct point-in-time grid, the grid ALONE scores 4.13 vs the engine's 2.26 and
the best blend weight is ZERO. CLAUDE.md's "never copy the grid" is correct.

The grid's value is as a MODEL FEATURE, not an anchor: the model learns when the
cell is informative and when to discount it. Measured across four out-of-time
splits, adding it as a feature is worth -0.27 to -0.63 MAE and cuts the >=5pt tail
by ~40% (e.g. 19.6% -> 11.9%); bootstrap +0.397, 95% CI [+0.340, +0.455].

This module answers exactly one question, honestly:
    "what did this stone's cell read on date D, using only edits made before D?"
"""

from __future__ import annotations

import bisect
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from ..config import DATA_DIR
from .master_grid import canon_shape, _FLUOR

log = logging.getLogger(__name__)

GRID_HISTORY = DATA_DIR / "master_grid" / "history.json"

# A cell edit older than this is still USED (it is the cell's live value until
# re-edited) but reported as stale so the caller can widen its interval.
STALE_AFTER_DAYS = 30


def _cell_key(shape, cell_id: str) -> str:
    return f"{str(shape).upper().strip()}|{cell_id}"


class GridHistory:
    """As-of lookup over the client's grid edit history.

    Storage shape: {"SHAPE|cellId": [[iso_date, discount], ...]} ascending by date.
    `cellId` is 'minW,maxW,color,clarity,cut,fluor' and does NOT contain the shape,
    so the shape is part of the key (otherwise sheets collide silently).
    """

    def __init__(self, raw: dict[str, list]):
        # index: (canon_shape, color, clarity, cut, fluor) -> [(minW, maxW, dates, discounts)]
        self._idx: dict[tuple, list] = {}
        n = 0
        for key, versions in raw.items():
            shape_tok, _, cid = key.partition("|")
            shape = canon_shape(shape_tok)
            if shape is None:
                continue                      # junk token (lab name / "NONE") — never index
            parts = cid.split(",")
            if len(parts) != 6:
                continue
            try:
                lo, hi = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            versions = sorted(versions, key=lambda v: v[0])
            dates = [str(v[0])[:10] for v in versions]
            discs = [float(v[1]) for v in versions]
            if not dates:
                continue
            k = (shape, parts[2].upper(), parts[3].upper(), parts[4].upper(), parts[5].upper())
            self._idx.setdefault(k, []).append((lo, hi, dates, discs))
            n += 1
        for v in self._idx.values():
            v.sort(key=lambda t: t[0])
        self.n_cells = n

    @classmethod
    def load(cls, path: Path | None = None) -> "GridHistory | None":
        p = path or GRID_HISTORY
        if not p.exists():
            log.warning("No grid history at %s — the grid feature will be unavailable "
                        "(run `python -m glowstar.ingestion.master_grid history`).", p)
            return None
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def as_of(self, shape, weight: float, color, clarity, cps, fluorescence,
              asof: str | date | datetime) -> tuple[float | None, int | None]:
        """(discount, age_days) for this stone's cell as of `asof`, else (None, None).

        Only edits STRICTLY BEFORE `asof` are visible, so this can never leak a
        value the desk had not yet written when the stone was priced.
        """
        sh = canon_shape(shape)
        if sh is None:
            return None, None
        if isinstance(asof, (date, datetime)):
            asof = asof.strftime("%Y-%m-%d")
        asof = str(asof)[:10]
        fl = _FLUOR.get(str(fluorescence or "").strip().upper(), "NON")
        key = (sh, str(color or "").strip().upper(), str(clarity or "").strip().upper(),
               str(cps or "").strip().upper(), fl)
        for lo, hi, dates, discs in self._idx.get(key, []):
            if lo <= float(weight) <= hi:
                i = bisect.bisect_left(dates, asof)
                if i == 0:
                    return None, None          # cell existed, but not yet on that date
                age = None
                try:
                    age = (datetime.strptime(asof, "%Y-%m-%d")
                           - datetime.strptime(dates[i - 1], "%Y-%m-%d")).days
                except ValueError:
                    pass
                return discs[i - 1], age
        return None, None


def attach_grid(df, history: "GridHistory | None", asof=None):
    """Add `grid_discount` / `grid_age_days` to `df` (point-in-time).

    `asof=None` uses each row's OWN OrderDate — correct for TRAINING, where every
    stone must see only the grid that existed when it sold. Pass an explicit date
    (e.g. today) when SERVING, where the current grid is legitimately available.

    Silent no-op (columns = NaN) when no history is loaded, so the engine still
    runs — it simply routes every stone to the non-grid model.
    """
    import numpy as np
    import pandas as pd

    out = df.copy()
    if history is None:
        out["grid_discount"] = np.nan
        out["grid_age_days"] = np.nan
        return out

    discs, ages = [], []
    for r in out.itertuples():
        when = asof if asof is not None else getattr(r, "OrderDate_dt", None)
        if when is None or (isinstance(when, float) and pd.isna(when)):
            discs.append(np.nan); ages.append(np.nan); continue
        d, a = history.as_of(getattr(r, "Shape_full", None), getattr(r, "Weight", 0.0),
                             getattr(r, "Color", None), getattr(r, "Clarity", None),
                             getattr(r, "CPS", None), getattr(r, "Fluorescence", None), when)
        discs.append(np.nan if d is None else d)
        ages.append(np.nan if a is None else a)
    out["grid_discount"] = discs
    out["grid_age_days"] = ages
    hit = float(pd.Series(discs).notna().mean()) if len(discs) else 0.0
    log.info("Grid feature attached: %.1f%% of %d stones have a point-in-time cell.",
             hit * 100, len(out))
    return out


def bank_history(days: int = 130, path: Path | None = None, chunk_days: int = 3) -> dict:
    """Pull grid edits and MERGE them into the point-in-time history store.

    Incremental: a daily run only needs a short window; the store accumulates. The
    first run needs a long window to cover the training period. ALL sheets are kept
    (never filter to "Master": measured over one window, that filter dropped 99.8%
    of CUSHION cells, 81% of PEAR and 61% of HEART).
    """
    from ..ingestion.diamanto import get_cells_history, get_access_token

    p = path or GRID_HISTORY
    p.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, list] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    before = sum(len(v) for v in store.values())

    token = get_access_token()
    today = date.today()
    cur = today - timedelta(days=days)
    seen: dict[str, set] = {}
    while cur < today:
        end = min(cur + timedelta(days=chunk_days), today)
        try:
            rows = get_cells_history(cur.isoformat(), end.isoformat(), token=token)
        except Exception:
            token = get_access_token()          # token may have expired mid-pull
            try:
                rows = get_cells_history(cur.isoformat(), end.isoformat(), token=token)
            except Exception:
                log.exception("grid history chunk %s..%s failed; continuing.", cur, end)
                cur = end
                continue
        for r in rows:
            cid, d = r.get("cellId"), r.get("discount")
            if not cid or d is None:
                continue
            cd = str(r.get("createdDate", ""))
            for shp in (r.get("shape") or []):
                k = _cell_key(shp, cid)
                if k not in seen:
                    seen[k] = {v[0] for v in store.get(k, [])}
                if cd in seen[k]:
                    continue                    # idempotent: same edit already stored
                store.setdefault(k, []).append([cd, float(d)])
                seen[k].add(cd)
        log.info("  grid history %s..%s: %d edits (store: %d cells)", cur, end, len(rows), len(store))
        cur = end

    for v in store.values():
        v.sort(key=lambda x: x[0])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)
    after = sum(len(v) for v in store.values())
    log.info("Banked grid history: %d cells, %d versions (+%d new) -> %s",
             len(store), after, after - before, p)
    return {"cells": len(store), "versions": after, "added": after - before}
