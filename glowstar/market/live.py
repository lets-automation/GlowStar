"""Live market layer: real-time Uni comparables per stone, cleaned.

For a stone, query the live Uni feed for matching listings (shape / size bracket
/ color / clarity / lab), clean them through the authenticity pipeline, and
return the market median discount + count + the cleaned comparables. Aggregating
these gives a LIVE MarketTables (segment medians + BGM deltas) the engine can use
instead of the banked snapshot — fully real-time, no hardcoded data.

Calls are cached per segment so pricing a book of stones doesn't re-query shared
segments. Requires Uni credentials (.env) and the verified codebook
(market.calibrate_codebook).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ..ingestion import uni
from .aggregate_bulk import _milky_severity, _shade_class, _median_from_hist
from .anchor import MarketTables
from .authenticity import clean_market_stones, CleanResult
from .mappings import CONFIRMED, UnmappedCodeError
from .segments import SIZE_EDGES, segment_keys, size_band

log = logging.getLogger(__name__)

# records.json fluorescence abbreviations -> Uni letter code key.
_FLUOR_TO_UNI = {
    "NON": "N", "NONE": "N", "FNT": "F", "FAINT": "F", "VSL": "VSL",
    "SLT": "SL", "SLIGHT": "SL", "MED": "M", "MEDIUM": "M",
    "STG": "S", "STRONG": "S", "VSTG": "VS",
}


def fluorescence_code(raw: str | None) -> int | None:
    """Uni fluorescence_intensity code for a stone's fluorescence (verified codebook)."""
    if not raw:
        return None
    letter = _FLUOR_TO_UNI.get(str(raw).strip().upper())
    return CONFIRMED.get("fluorescence", {}).get(letter) if letter else None


def _tight_window(weight: float, delta: float) -> tuple[float, float]:
    """A small size window around the weight, CLAMPED to the stone's Rap bracket
    so we respect the bracket price cliff and keep the response small/relevant."""
    b = size_band(weight)
    blo = SIZE_EDGES[b]
    bhi = SIZE_EDGES[b + 1] - 0.01 if b + 1 < len(SIZE_EDGES) else weight + 1.0
    lo = max(blo, round(weight - delta, 2))
    hi = min(bhi, round(weight + delta, 2))
    return lo, hi


class LiveMarket:
    """Live Uni comparables with per-segment caching."""

    def __init__(self, max_age_days: int = 180, match_lab: bool = True):
        self.max_age_days = max_age_days
        self.match_lab = match_lab
        self._cache: dict[tuple, CleanResult] = {}
        self.calls = 0

    def _query(self, shape, lo, hi, color, clarity, lab, fl_code):
        """(result, queryable). queryable=False means an unmapped code; result is
        None on a network/timeout failure. Filters on the price-relevant factors
        so the response is small, fast, and genuinely comparable."""
        try:
            body = uni.build_filter(
                shape=shape, size_from=lo, size_to=hi, colors=[color], clarities=[clarity],
                lab=(lab if self.match_lab and lab in ("GIA", "IGI", "HRD") else None),
                fluorescence_code=fl_code,
            )
        except UnmappedCodeError:
            return None, False
        try:
            raw = uni.fetch_market(body)
            self.calls += 1
            return clean_market_stones(raw, max_age_days=self.max_age_days), True
        except Exception as e:
            log.warning("live Uni pull failed (%.2f-%.2f %s/%s): %s", lo, hi, color, clarity, str(e)[:60])
            return None, True

    def comparables(self, shape: str, weight: float, color: str, clarity: str,
                    lab: str | None = None, fluorescence: str | None = None) -> CleanResult | None:
        """Live cleaned comparables matched on the price-relevant factors: tight
        size window (clamped to the Rap bracket), color, clarity, lab, AND
        fluorescence. This pulls only the NECESSARY, genuinely-comparable stones
        — small/fast (no huge dump that times out) and more accurate. If it still
        fails, widens the size window once before giving up.
        """
        fl_code = fluorescence_code(fluorescence)
        lo, hi = _tight_window(weight, delta=0.05)
        key = (shape, round(lo, 2), round(hi, 2), color, clarity,
               lab if self.match_lab else None, fl_code)
        if key in self._cache:
            return self._cache[key]

        res, queryable = self._query(shape, lo, hi, color, clarity, lab, fl_code)
        if (res is None or res.report.n_used < 3) and queryable:
            # Too few / failed: widen the size window once (still within bracket).
            wlo, whi = _tight_window(weight, delta=0.12)
            res2, _ = self._query(shape, wlo, whi, color, clarity, lab, fl_code)
            if res2 is not None and (res is None or res2.report.n_used > res.report.n_used):
                res = res2
        self._cache[key] = res
        return res

    def build_tables(self, stones, base: MarketTables | None = None) -> MarketTables:
        """Build a LIVE MarketTables (segment medians + BGM deltas) for a set of
        stones. Falls back to `base` (e.g. the banked artifact) for segments with
        no live comparables, so the engine always has a market reference.

        The anchor median uses the NO-BGM (clean) comparables only (client's
        "BGM as base"): a BGM-affected comp shouldn't drag the clean base price
        down. Falls back to the all-comps median if a segment has too few clean
        comps."""
        segments: dict[str, dict] = dict(base.segments) if base else {}
        pooled: list[dict] = []

        total = len(stones)
        for i, st in enumerate(stones.itertuples(), 1):
            if i % 10 == 0 or i == 1:
                log.info("live comparables: stone %d/%d (%d Uni calls so far)", i, total, self.calls)
            res = self.comparables(st.Shape_full, st.Weight, st.Color, st.Clarity,
                                   st.Lab, getattr(st, "Fluorescence", None))
            if not res or res.report.median_discount is None:
                continue
            pooled.extend(res.stones)
            clean_med, clean_n = no_bgm_median(res.stones)
            anchor_med = clean_med if clean_n >= 5 else res.report.median_discount
            name = "|".join(map(str, segment_keys(st.Shape_full, st.Weight, st.Color, st.Clarity)[0]))
            segments[name] = {
                "median_discount": round(anchor_med, 2), "n": res.report.n_used,
                "no_bgm_median": None if clean_med is None else round(clean_med, 2),
                "no_bgm_n": clean_n,
                "bgm_free_share": round(clean_n / max(1, res.report.n_used), 3),
                "source": "live",
            }

        bgm = self._bgm_from_pool(pooled) if pooled else (base.bgm if base else _empty_bgm())
        return MarketTables(segments=segments, bgm=bgm)

    @staticmethod
    def _bgm_from_pool(stones: list[dict]) -> dict:
        by_milky: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        by_shade: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        clean: dict[int, int] = defaultdict(int)
        for s in stones:
            d = s.get("discount")
            if d is None:
                continue
            b = int(round(d))
            m = _milky_severity(s.get("milky"))
            sh = _shade_class(s.get("shade"))
            by_milky[m][b] += 1
            by_shade[sh][b] += 1
            if m == "none" and sh in ("none", "neutral"):
                clean[b] += 1
        ref = _median_from_hist(clean)[0]
        ref = ref if not np.isnan(ref) else 0.0

        def deltas(table):
            out = {}
            for lvl, hist in table.items():
                med, n = _median_from_hist(hist)
                out[lvl] = {"median_discount": med, "n": n,
                            "delta_vs_clean": round((med - ref), 2) if not np.isnan(med) else 0.0}
            return out

        return {"reference_clean_median": ref, "by_milky": deltas(by_milky),
                "by_shade": deltas(by_shade), "source": "live"}


def no_bgm_median(stones: list[dict]) -> tuple[float | None, int]:
    """Median discount of the NO-BGM (clean) comparables — the clean base price.

    Clean = is_bgm != Yes AND not milky AND shade not value-negative. Uses the
    is_bgm/milky/shade fields the Uni feed carries on every stone.
    """
    clean = [s["discount"] for s in stones
             if s.get("discount") is not None
             and str(s.get("is_bgm")) != "Yes"
             and _milky_severity(s.get("milky")) == "none"
             and _shade_class(s.get("shade")) in ("none", "neutral")]
    if not clean:
        return None, 0
    return float(np.median(clean)), len(clean)


def _empty_bgm() -> dict:
    return {"reference_clean_median": 0.0, "by_milky": {}, "by_shade": {}, "source": "none"}
