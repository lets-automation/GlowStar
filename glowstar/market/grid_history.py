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

# Cut tiers as the client's grid publishes them, best -> worst. Used ONLY by the
# cut-tier backoff below; the ordering is what makes "nearest published tier"
# meaningful.
_CUT_ORDER: tuple[str, ...] = ("3EX", "EX", "VG-GD", "VG", "GD", "FR")

# Match quality of a returned cell. Exposed so a caller can tell an exact cell
# from a substituted one — the grid is a FEATURE, and how much to trust it
# depends on how it was found.
MATCH_EXACT = 0
MATCH_CUT_BACKOFF = 1


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

    def _read_key(self, key: tuple, weight: float, asof: str):
        """(discount, age_days) from one exact key, requiring a bracket that
        CONTAINS the weight. None when the key is absent, the weight falls outside
        every published bracket, or no containing bracket has an edit before `asof`.

        THE NARROWEST CONTAINING BRACKET DECIDES (ties: lower `lo`, then the
        earlier-sorted one). The client's grid publishes OVERLAPPING brackets for
        the same cell: the Master sheet carries narrow slots (0.35-0.39, 0.63-0.64)
        AND coarse 0.x0-0.x9 overlays, and two non-Master sheets add 0.09-wide
        ROUND cells that share this key space because the key drops the sheet name.

        MEASURED 2026-09-02 against realized sales (18,939 sold since 2026-03-15,
        grid read as of the day before each sale):

            rule                              n      grid-vs-sale MAE   within5
            first containing bracket      18,075        4.94             72.1%
            narrowest containing bracket  18,075        3.35             80.9%
            conflict stones only, first    2,127       16.74              9.7%
            conflict stones only, narrow   2,127        3.17             85.0%

        All 2,127 conflict stones are Round. Through the shipped engine, paired on
        7,255 stones over six weekly origins: MAE 1.857 -> 1.611 (95% CI on the
        delta [-0.279, -0.215]), >=5pt tail 6.6% -> 4.7%, better on every origin;
        fancies and non-conflict rounds moved by refit noise only (<0.03).
        Confirmed on production: 10 of the 18 worst served-quote misses on
        2026-09-03 were the coarse overlay winning over the desk's own narrow slot.

        A bracket with no edit before `asof` is skipped, not returned as None —
        the next-narrowest edited bracket answers instead.
        """
        best = None                             # (width, lo, discount, edit_date)
        for lo, hi, dates, discs in self._idx.get(key, []):
            if lo <= weight <= hi:
                i = bisect.bisect_left(dates, asof)
                if i == 0:
                    continue                    # bracket exists, but not yet edited
                cand = (hi - lo, lo, discs[i - 1], dates[i - 1])
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best is None:
            return None, None
        age = None
        try:
            age = (datetime.strptime(asof, "%Y-%m-%d")
                   - datetime.strptime(best[3], "%Y-%m-%d")).days
        except ValueError:
            pass
        return best[2], age

    def as_of_detailed(self, shape, weight: float, color, clarity, cps, fluorescence,
                       asof: str | date | datetime
                       ) -> tuple[float | None, int | None, int | None]:
        """(discount, age_days, match_level) for this stone's cell as of `asof`.

        Only edits STRICTLY BEFORE `asof` are visible, so this can never leak a
        value the desk had not yet written when the stone was priced.

        MATCH_CUT_BACKOFF — WHY THIS EXISTS, AND WHY ONLY FOR FANCIES
        ------------------------------------------------------------
        The client's grid publishes the `FR` cut for ROUND ONLY. Every fancy sheet
        carries just 3EX/EX/VG-GD/VG/GD, so a fancy stone whose CPS is a tier the
        sheet omits has NO cell at all — not because the desk has no view on it,
        but because that one coordinate is unpublished. Missing the cell is the
        expensive outcome: measured out-of-time, a fancy stone WITH a cell scores
        2.73 MAE and one without scores 8.60, with 63% of them >=5 points out.

        Measured on 17,748 sales (2026-03-15 onward), grid cell vs the realized
        discount, for FANCY shapes:
            exact cell            n=5198   MAE 4.26   within5 72.8%
            cut-tier backoff      n= 140   MAE 4.43   within5 76.4%   <- as good
            fluorescence backoff  n=  13   MAE 12.48  within5  7.7%   <- REFUSED
            nearest bracket       n= 182   MAE  9.75  within5 36.8%   <- REFUSED
        So the substituted-cut cell is statistically as informative as a real one,
        while the other two substitutions are actively misleading and are NOT done.

        ROUND IS DELIBERATELY EXCLUDED. GIA grades overall CUT for round brilliants
        and the market prices it hard, so swapping the cut tier swaps a genuinely
        different stone — and the numbers agree: round cut-backoff scored MAE 8.69
        against 5.20 for an exact cell. `segments.cut_graded()` already encodes
        exactly this distinction; this uses it rather than inventing a second rule.

        This is STRICTLY ADDITIVE. When an exact cell exists the answer is
        byte-identical to before, so the 99.4% of rounds and 92.4% of fancies that
        already resolved are untouched.
        """
        sh = canon_shape(shape)
        if sh is None:
            return None, None, None
        if isinstance(asof, (date, datetime)):
            asof = asof.strftime("%Y-%m-%d")
        asof = str(asof)[:10]
        w = float(weight)
        fl = _FLUOR.get(str(fluorescence or "").strip().upper(), "NON")
        co = str(color or "").strip().upper()
        cl = str(clarity or "").strip().upper()
        cu = str(cps or "").strip().upper()

        d, age = self._read_key((sh, co, cl, cu, fl), w, asof)
        if d is not None:
            return d, age, MATCH_EXACT

        # Cut-tier backoff — fancies only, and only when the stone's own tier is a
        # known one. Nearest published tier by rank distance; ties prefer the
        # BETTER tier, because the grid omits the worst tiers far more often than
        # the best and a one-step-better cell is the closer comparable.
        from .segments import cut_graded
        if not cut_graded(shape) and cu in _CUT_ORDER:
            i = _CUT_ORDER.index(cu)
            for j in sorted(range(len(_CUT_ORDER)), key=lambda j: (abs(j - i), j)):
                if j == i:
                    continue
                d, age = self._read_key((sh, co, cl, _CUT_ORDER[j], fl), w, asof)
                if d is not None:
                    return d, age, MATCH_CUT_BACKOFF
        return None, None, None

    def as_of(self, shape, weight: float, color, clarity, cps, fluorescence,
              asof: str | date | datetime) -> tuple[float | None, int | None]:
        """(discount, age_days) — see `as_of_detailed`. Kept for existing callers."""
        d, age, _ = self.as_of_detailed(shape, weight, color, clarity, cps,
                                        fluorescence, asof)
        return d, age


def attach_grid(df, history: "GridHistory | None", asof=None,
                date_col: str = "OrderDate_dt"):
    """Add `grid_discount` / `grid_age_days` to `df` (point-in-time).

    `asof=None` uses each row's own `date_col` — correct for TRAINING, where every
    stone must see only the grid that existed when it sold. Pass an explicit date
    (e.g. today) when SERVING, where the current grid is legitimately available.

    `date_col` selects WHICH per-row date is "point-in-time". It defaults to
    `OrderDate_dt` (the sale date) because that is what PRICING trains on. The
    VELOCITY model needs a different one: predicting time-to-sell AT LISTING may
    only see the grid as it stood on `MarketSheetDate_dt`, so
    `glowstar.inventory.survival` passes that instead. Using the sale-date grid to
    predict the time to that same sale is leakage of exactly the kind CLAUDE.md
    documents.

    Silent no-op (columns = NaN) when no history is loaded, so the engine still
    runs — it simply routes every stone to the non-grid model.
    """
    import numpy as np
    import pandas as pd

    out = df.copy()
    if history is None:
        out["grid_discount"] = np.nan
        out["grid_age_days"] = np.nan
        out["grid_match_level"] = np.nan
        return out

    discs, ages, levels = [], [], []
    for r in out.itertuples():
        when = asof if asof is not None else getattr(r, date_col, None)
        if when is None or (isinstance(when, float) and pd.isna(when)):
            discs.append(np.nan); ages.append(np.nan); levels.append(np.nan); continue
        d, a, lv = history.as_of_detailed(
            getattr(r, "Shape_full", None), getattr(r, "Weight", 0.0),
            getattr(r, "Color", None), getattr(r, "Clarity", None),
            getattr(r, "CPS", None), getattr(r, "Fluorescence", None), when)
        discs.append(np.nan if d is None else d)
        ages.append(np.nan if a is None else a)
        levels.append(np.nan if lv is None else float(lv))
    out["grid_discount"] = discs
    out["grid_age_days"] = ages
    # Diagnostic only — NOT a model feature. Adding it to GRID_FEATURES would
    # change the grid model's matrix width, and every engine already pickled in
    # the registry would then predict against a schema it was not fit with. That
    # is the same defect `baseline._fitted_levels` exists to prevent; if this ever
    # becomes a feature it needs the same guard.
    out["grid_match_level"] = levels
    s = pd.Series(discs)
    hit = float(s.notna().mean()) if len(discs) else 0.0
    n_backoff = int(pd.Series(levels).eq(float(MATCH_CUT_BACKOFF)).sum())
    log.info("Grid feature attached: %.1f%% of %d stones have a point-in-time cell"
             "%s.", hit * 100, len(out),
             f" ({n_backoff} via cut-tier backoff)" if n_backoff else "")
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
