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
from .segments import (SIZE_EDGES, cut_graded, cut_tier, segment_keys, size_band,
                       size_bucket_window, size_tag)

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


def _demand_decade(weight: float) -> tuple[float, float]:
    """The 0.10ct size 'decade' a stone sits in (e.g. 0.83 -> 0.80-0.89).

    Rapaport $/ct is FLAT across a bracket (0.70-0.89), but market DEMAND is not:
    it steps at the round-tenth thresholds (0.80, 0.90, 1.00 ...). A 0.83ct stone
    (above the 0.80 cutoff) lists materially shallower than a 0.72ct one; mixing
    them drags the comp median deep. Comparable windows must never cross the
    threshold the stone sits above — verified: 0.71-0.79 J/VVS2 = -45 vs 0.80+ =
    -40, so a 0.71-0.88 pull mis-anchored a 0.83 stone ~5 pts too deep."""
    import math
    lo = math.floor(round(weight, 4) * 10 + 1e-9) / 10.0
    return round(lo, 2), round(lo + 0.09, 2)


def _tight_window(weight: float, delta: float) -> tuple[float, float]:
    """A small size window around the weight, CLAMPED to the stone's Rap bracket
    so we respect the bracket price cliff and keep the response small/relevant.

    NOTE: the pull stays WIDE on purpose (comp support). The demand-threshold
    restriction (0.80/0.90/1.00) is applied later to the MEDIAN comps only
    (`_build_one`), so tightening never starves the segment into a bad fallback."""
    b = size_band(weight)
    blo = SIZE_EDGES[b]
    bhi = SIZE_EDGES[b + 1] - 0.01 if b + 1 < len(SIZE_EDGES) else weight + 1.0
    lo = max(blo, round(weight - delta, 2))
    hi = min(bhi, round(weight + delta, 2))
    return lo, hi


class LiveMarket:
    """Live Uni comparables with per-segment caching."""

    def __init__(self, max_age_days: int = 180, match_lab: bool = True,
                 min_segment_n: int = 12, size_tier: bool = True):
        self.max_age_days = max_age_days
        self.match_lab = match_lab
        # Compute the anchor median from comps in the stone's OWN size-decade
        # (respecting the 0.80/0.90/1.00 demand thresholds) when well-supported.
        # Off = legacy behaviour (median over the whole pulled window).
        self.size_tier = size_tier
        # Minimum live cut-tier comps to PUBLISH a live segment over the banked
        # cut-aware aggregate (matches the anchor's market_median min_n).
        self.min_segment_n = min_segment_n
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
                    lab: str | None = None, fluorescence: str | None = None,
                    full_bracket: bool = False, size_bucket: bool = False) -> CleanResult | None:
        """Live cleaned comparables matched on the price-relevant factors: size,
        color, clarity, lab, AND fluorescence.

        `full_bracket=True` queries the stone's WHOLE Rap size bracket (lo..hi of
        the band), not a tight ±0.05 window. This matters for the cut-tier split:
        a tight window starves the minority cut tier (e.g. only a few VG comps),
        so the VG median collapses to the cut-BLIND, EX-dominated blend and a VG
        stone is mis-anchored ~8-10 pts too shallow. The full bracket gives the
        cut tier real support (verified: 0.80 D/VS2 tight=33 comps/<5 VG -> blend;
        full bracket=200 VG comps at the true VG level). Used for segment medians.
        The tight window stays the default for a per-stone "very close comps" panel.
        """
        fl_code = fluorescence_code(fluorescence)
        if size_bucket:
            # ROUND -> the client's exact price slot (0.84 -> 0.83-0.84, distinct
            # from 0.85-0.89); other shapes -> a 0.10ct sub-bucket. Either way the
            # market is pulled size-locally, not lumped across the whole bracket.
            lo, hi = size_bucket_window(weight, shape)
        elif full_bracket:
            b = size_band(weight)
            blo = SIZE_EDGES[b]
            bhi = round(SIZE_EDGES[b + 1] - 0.01, 2) if b + 1 < len(SIZE_EDGES) else round(weight + 1.0, 2)
            if bhi - blo <= 0.30:
                lo, hi = blo, bhi                     # narrow bracket: take it whole
            else:
                # WIDE bracket (e.g. 1.00-1.49): the full span returns a huge,
                # timeout-prone response. Rap $/ct is flat within a bracket, so a
                # 0.30-wide window centred on the stone stays price-comparable and
                # still carries ample cut-tier support (verified: 1.00-1.15 -> ~20-60
                # VG comps), while the full 1.00-1.49 pull times out.
                lo = max(blo, round(weight - 0.15, 2))
                hi = min(bhi, round(weight + 0.15, 2))
        else:
            lo, hi = _tight_window(weight, delta=0.05)
        key = (shape, round(lo, 2), round(hi, 2), color, clarity,
               lab if self.match_lab else None, fl_code)
        if key in self._cache:
            return self._cache[key]

        res, queryable = self._query(shape, lo, hi, color, clarity, lab, fl_code)
        if (res is None or res.report.n_used < 3) and queryable and not full_bracket:
            # Too few / failed: widen the size window once (still within bracket).
            wlo, whi = _tight_window(weight, delta=0.12)
            res2, _ = self._query(shape, wlo, whi, color, clarity, lab, fl_code)
            if res2 is not None and (res is None or res2.report.n_used > res.report.n_used):
                res = res2
        # Cache ONLY successes. A transient failure (timeout/throttle under batch
        # load) must be retried later, not pinned to None for the rest of the run.
        if res is not None:
            self._cache[key] = res
        return res

    def build_tables(self, stones, base: MarketTables | None = None,
                     max_passes: int = 3) -> MarketTables:
        """Build a LIVE MarketTables (segment medians + BGM deltas) for a set of
        stones. Falls back to `base` (e.g. the banked artifact) for segments with
        no live comparables, so the engine always has a market reference.

        Resilient to throttling: a stone whose live pull FAILS (timeout/throttle
        under batch load) is retried in a later pass — not silently dropped to a
        coarse fallback after one miss. The anchor median uses the NO-BGM (clean)
        comparables only (client's "BGM as base"). Genuinely thin segments fall
        back to the banked cut-aware median (not retried — that is data, not a
        transient failure)."""
        segments: dict[str, dict] = dict(base.segments) if base else {}
        pooled: list[dict] = []

        pending = list(stones.itertuples())
        for pass_no in range(1, max_passes + 1):
            failed = []
            for i, st in enumerate(pending, 1):
                if i % 10 == 0 or i == 1:
                    log.info("live comparables: pass %d, stone %d/%d (%d calls)",
                             pass_no, i, len(pending), self.calls)
                if not self._build_one(st, segments, pooled):
                    failed.append(st)
            if not failed:
                break
            log.info("live market: %d segment(s) failed pass %d — retrying.", len(failed), pass_no)
            pending = failed

        bgm = self._bgm_from_pool(pooled) if pooled else (base.bgm if base else _empty_bgm())
        fluor = self._fluor_from_pool(pooled) if pooled else (getattr(base, "fluor", {}) if base else {})
        return MarketTables(segments=segments, bgm=bgm, fluor=fluor)

    @staticmethod
    def _fluor_from_pool(stones: list[dict]) -> dict:
        """Market fluorescence delta per (colour-group, level): how much deeper the
        market lists Faint/Medium/Strong vs None. Colour-grouped because the penalty
        is real for near-colourless (D-H) but ~neutral for lower colours."""
        from ..reference.normalize import normalize_fluorescence
        NC = set("DEFGH")
        hist: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for s in stones:
            d = s.get("discount")
            if d is None:
                continue
            fl = normalize_fluorescence(s.get("fluorescence"))
            c = str(s.get("color") or "").strip().upper()[:1]
            g = "nc" if c in NC else "low"
            hist[(g, fl)][int(round(d))] += 1
        by_group: dict[str, dict] = {}
        for g in ("nc", "low"):
            ref, refn = _median_from_hist(hist.get((g, "None"), {}))
            if np.isnan(ref) or refn < 20:            # need a stable None-fluoro baseline
                continue
            for fl in ("Faint", "Medium", "Strong", "Very Strong"):
                med, n = _median_from_hist(hist.get((g, fl), {}))
                if not np.isnan(med) and n >= 10:     # and enough of that level
                    by_group[f"{g}|{fl}"] = {"delta_vs_none": round(med - ref, 2), "n": n}
        return {"by_group": by_group, "source": "live"}

    def _build_one(self, st, segments: dict, pooled: list) -> bool:
        """Resolve one stone's live segment into `segments`. Returns False ONLY on
        a transient pull failure (caller retries it); True once handled (live
        segment written, or genuinely thin so the banked value stands)."""
        # SIZE-LOCAL: pull the stone's 0.10ct sub-bucket (not the whole bracket),
        # so an upper-bracket stone isn't anchored to deeper smaller-size comps.
        res = self.comparables(st.Shape_full, st.Weight, st.Color, st.Clarity,
                               st.Lab, fluorescence=None, size_bucket=True)
        if not res or res.report.median_discount is None:
            return False                  # transient failure -> retry this stone
        pooled.extend(res.stones)
        # CUT-AWARE: match the stone to its OWN cut tier (VG trades ~10 pts deeper
        # than EX/3EX; the EX-dominated blend over-prices it).
        tier = cut_tier(getattr(st, "CPS", None))
        tier_stones = [s for s in res.stones if cut_tier(s.get("cut")) == tier]
        # A cut-graded round whose own tier is thin even at full bracket: keep the
        # banked cut-AWARE median rather than collapsing to the cut-blind blend.
        if cut_graded(st.Shape_full) and len(tier_stones) < 8:
            return True                   # data-thin, not a failure
        if len(tier_stones) < 5:
            tier_stones = res.stones
        # Key the live segment by its size sub-slot (client's ROUND price slots;
        # 0.10ct buckets for other shapes) so different sizes in the same Rap
        # bracket don't collide. Banked (bracket-level) keys remain the fallback.
        name = "|".join(map(str, segment_keys(st.Shape_full, st.Weight, st.Color,
                            st.Clarity, getattr(st, "CPS", None))[0])) + size_tag(st.Weight, st.Shape_full)
        # A thin live segment must not shadow a better-supported banked one.
        if len(tier_stones) < self.min_segment_n and name in segments:
            return True                   # data-thin, not a failure
        # SIZE-TIER: within a Rap bracket, market demand steps at the round-tenth
        # thresholds (0.80/0.90/1.00). Take the median from comps in the stone's
        # OWN size-decade when well-supported (>=8), so a 0.83 stone isn't dragged
        # deep by sub-0.80 comps (verified: 0.71-0.79 J/VVS2=-45 vs 0.80+=-40).
        # The wide pull still provides support; only the median narrows. Falls back
        # to the full tier when the decade is thin, so it never starves.
        med_stones = tier_stones
        if self.size_tier:
            dlo, dhi = _demand_decade(st.Weight)
            dec = [s for s in tier_stones if s.get("size") is not None
                   and dlo <= float(s["size"]) <= dhi]
            if len(dec) >= 8:
                med_stones = dec
        tier_discs = [s["discount"] for s in med_stones if s.get("discount") is not None]
        all_med = float(np.median(tier_discs)) if tier_discs else res.report.median_discount
        clean_med, clean_n = no_bgm_median(med_stones)
        anchor_med = clean_med if clean_n >= 5 else all_med
        segments[name] = {
            "median_discount": round(anchor_med, 2), "n": len(tier_stones),
            "no_bgm_median": None if clean_med is None else round(clean_med, 2),
            "no_bgm_n": clean_n,
            "bgm_free_share": round(clean_n / max(1, len(tier_stones)), 3),
            "cut_tier": tier, "source": "live",
        }
        return True

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
