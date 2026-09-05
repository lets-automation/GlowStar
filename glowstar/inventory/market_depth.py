"""Market depth — the OTHER number, kept deliberately apart from velocity.

MOU 5.2 is explicit and MOU 10.3 makes it an acceptance condition: *a stone not
selling in the broad market does not mean it isn't selling for THEM.* A segment
can be illiquid market-wide and still move fast through Glow Star's own channel.
The system reports both figures and their ratio, separately, and never blends
them into one headline gauge however much better one would look on a dashboard.

So this module answers only "how deep is the broad market here?" and knows
nothing about how fast the desk sells. `velocity.py` answers only "how fast do
WE sell it" and never sees a depth number. `own_vs_market()` puts the two side
by side and names the gap — which is the client's edge, and the reason
Workstream B exists at all.

WHAT DEPTH IS AND IS NOT
------------------------
It is a COUNT of genuine comparable listings, after `market/authenticity.py`
removes the duplicate "virtual inventory" that inflates every raw market count.
It is NOT a price level: the feed is an ASKING market roughly 6 points shallower
than where this desk actually sells, and blending it toward our price was
measured monotonically harmful (`anchor_lambda` ships at 0.0). Depth and
liquidity only — CLAUDE.md Trap 8.7.

A FAILED LOOKUP IS `None`, NEVER `0`
------------------------------------
Zero comparables and "we could not reach the feed" are opposite findings: zero
means the desk is alone in the segment, which scores WELL. Conflating them would
hand the desk a glowing competition score every time the market API times out.
Every function here returns None plus a basis string on failure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Depth scoring. Log-scaled because the difference between 5 and 50 comparable
# listings matters far more than 5,000 vs 5,050 — the same shape `ai_score`
# already uses, so the desk sees one consistent notion of "deep".
_DEPTH_FULL = 400.0        # comps at which the segment scores 100 for depth


@dataclass
class DepthResult:
    """One segment's market depth, with the basis it rests on."""

    depth: int | None = None
    score: int | None = None
    basis: str = "not looked up"

    def as_dict(self) -> dict:
        return {"market_depth": self.depth, "market_depth_score": self.score,
                "market_depth_basis": self.basis}


def depth_score(depth: int | None) -> int | None:
    """0-100 where 100 is a deep, liquid segment. None stays None."""
    if depth is None:
        return None
    n = max(0, int(depth))
    if n == 0:
        return 0
    return int(max(0, min(100, round(100.0 * math.log10(1.0 + n)
                                     / math.log10(1.0 + _DEPTH_FULL)))))


def depth_for(shape, weight: float, color, clarity, *, lab=None,
              fluorescence=None, market=None) -> DepthResult:
    """Live comparable count for one stone's segment, after the dedup.

    `full_bracket=True` matches how the pricing engine pulls segment-level
    comparables, so the count the desk reads here is the same population the
    price was reasoned about — not a second, tighter definition of "comparable"
    that happens to disagree.
    """
    if market is None:
        return DepthResult(basis="no market source supplied")
    try:
        res = market.comparables(shape, float(weight), color, clarity,
                                 lab=lab, fluorescence=fluorescence,
                                 full_bracket=True)
    except Exception:
        log.exception("market depth lookup failed for %s %s %s %s",
                      shape, weight, color, clarity)
        return DepthResult(basis="market feed unavailable (lookup failed)")
    if res is None:
        return DepthResult(basis="market feed returned nothing for this segment")
    n = int(getattr(res.report, "n_used", 0) or 0)
    raw = int(getattr(res.report, "n_in", 0) or 0)
    return DepthResult(
        depth=n, score=depth_score(n),
        basis=(f"{n:,} genuine comparable listings"
               + (f" (from {raw:,} raw, after the duplicate-listing dedup)" if raw > n else "")))


def banked_depth_table(stones, *, path=None, segment_col: str = "segment",
                       max_age_days: float = 45.0) -> "DepthTable":
    """Depth for the WHOLE book from the banked market artifact.

    WHY THIS EXISTS RATHER THAN THE LIVE PULL
    ------------------------------------------
    `build_depth_table()` asks the live feed once per segment. Measured on this
    book: **44 seconds per segment across 1,958 distinct stock segments — about
    24 hours** for one pass, with 1 in 5 lookups failing. That is not a slow
    path, it is an unusable one, and MOU 5.2's own-vs-market ratio is an
    acceptance condition under 10.3 — so leaving depth null across the book was
    not an option either.

    `artifacts/market_segments.json` already holds per-segment comparable counts
    at four granularities, built by `market/aggregate_bulk.py` from the Uni bulk
    dump with the same dedup-by-certificate that removes duplicate "virtual
    inventory". Keys are `Shape|size_band|Color|Clarity`, which is exactly the
    survival frame's own segment key, so the join is direct and the whole book
    resolves instantly.

    THE TRADE, STATED: this is a BANKED snapshot, not live. Its age travels on
    every row's basis, and past `max_age_days` the basis says so in as many
    words. Depth is a structural property of a segment and moves far more slowly
    than a price does — but "slowly" is not "never", and a reader must be able
    to see how old the number is. Use `depth_for(..., market=LiveMarket())` when
    one specific segment needs a live answer.
    """
    import json
    import os
    from datetime import date

    from ..config import ARTIFACTS_DIR

    table = DepthTable()
    p = path or (ARTIFACTS_DIR / "market_segments.json")
    try:
        banked = json.loads(open(p, encoding="utf-8").read())
        age = (date.today() - date.fromtimestamp(os.path.getmtime(p))).days
    except Exception:
        log.exception("banked market segments unavailable at %s", p)
        return table
    if not len(stones):
        return table

    # SCORE BY PERCENTILE, NOT AGAINST A CONSTANT. `depth_score()` maps an
    # absolute count and is calibrated for the LIVE per-segment pull, whose
    # counts run in the tens-to-hundreds. The banked snapshot is a whole-market
    # bulk aggregate whose counts run to six figures (median 39, p90 813, p99
    # 7,792), so every real segment pinned at 100 and the own-vs-market label
    # read "liquid both ways" for essentially the entire book — a score with no
    # variance is not a score. Ranking each segment against the snapshot's own
    # distribution also makes depth directly comparable to the velocity score,
    # which is itself a percentile of this desk's own book.
    ref = sorted(int(v["n"]) for v in banked.values() if v.get("n"))
    def _pct(n: int) -> int:
        import bisect
        if not ref:
            return depth_score(n)
        return int(round(100.0 * bisect.bisect_left(ref, n) / len(ref)))

    stale = age > max_age_days
    age_note = (f"banked market snapshot, {age} days old"
                + ("  — STALE: past the "
                   f"{max_age_days:.0f}-day guide, refresh with "
                   "glowstar.market.aggregate_bulk" if stale else ""))

    segments = list(dict.fromkeys(stones[segment_col].astype(str)))
    table.n_requested = len(segments)
    for seg in segments:
        parts = seg.split("|")
        # Most specific first, then back off — the same hierarchy the pricing
        # engine uses, so a segment never silently answers at a level the report
        # does not name.
        res = None
        for depth_level in range(len(parts), 0, -1):
            key = "|".join(parts[:depth_level])
            hit = banked.get(key)
            if hit and hit.get("n"):
                n = int(hit["n"])
                where = ("this segment" if depth_level == len(parts) else
                         f"{key} — a COARSER segment; the exact one is not in "
                         f"the snapshot")
                sc = _pct(n)
                res = DepthResult(
                    depth=n, score=sc,
                    basis=(f"{n:,} genuine comparable listings in {where} — "
                           f"{sc}th percentile of market depth in the snapshot; "
                           f"{age_note}"))
                break
        if res is None:
            # Not a lookup failure — a COVERAGE limit, and the two must not read
            # the same. The banked artifact carries Round only (it was built
            # from a Round bulk pull, and the 6.2 GB source is no longer on
            # disk), so every fancy shape lands here. Saying "0 comparables"
            # would score them as "you are alone in this segment", which is the
            # best possible score for the worst possible reason.
            res = DepthResult(
                basis=(f"no market depth available for '{parts[0]}': the banked "
                       f"snapshot covers Round only. Re-run "
                       f"glowstar.market.aggregate_bulk over a bulk dump that "
                       f"includes fancy shapes, or pull this segment live."))
            table.n_failed += 1
        else:
            table.n_resolved += 1
        table.by_segment[seg] = res
    log.info("%s (banked, %d days old)", table.summary(), age)
    return table


@dataclass
class DepthTable:
    """Per-segment depth for a whole book, with its coverage stated.

    Depth is a LIVE lookup per segment, so a full book is thousands of API
    calls. The table therefore covers the segments that matter most and says
    plainly how many it reached: a report that quietly depth-scored 12% of the
    book and left the rest blank would read as "the market is thin everywhere".
    """

    by_segment: dict[str, DepthResult] = field(default_factory=dict)
    n_requested: int = 0
    n_resolved: int = 0
    n_failed: int = 0

    @property
    def coverage(self) -> float:
        return (self.n_resolved / self.n_requested) if self.n_requested else 0.0

    def get(self, segment: str) -> DepthResult:
        return self.by_segment.get(segment, DepthResult(basis="segment not looked up"))

    def summary(self) -> str:
        return (f"market depth: {self.n_resolved}/{self.n_requested} segments "
                f"resolved ({self.coverage:.0%}), {self.n_failed} failed")


def build_depth_table(stones, *, market=None, max_segments: int = 250,
                      segment_col: str = "segment") -> DepthTable:
    """Depth for the segments carrying the most stock, largest first.

    `stones` is a frame with `segment`, `shape`, `weight`, `color`, `clarity`
    (the survival frame's own columns). Segments are ranked by how many stones
    sit in them, so the cap falls on the tail the desk cares least about, and
    the coverage figure travels with the result.
    """
    import pandas as pd

    table = DepthTable()
    if market is None or not len(stones):
        return table
    counts = stones.groupby(segment_col, observed=True).size().sort_values(ascending=False)
    wanted = list(counts.index[:max_segments])
    table.n_requested = len(wanted)
    reps = (stones.drop_duplicates(subset=[segment_col])
            .set_index(segment_col))
    for seg in wanted:
        try:
            r = reps.loc[seg]
        except KeyError:
            continue
        res = depth_for(r["shape"], r["weight"], r["color"], r["clarity"],
                        lab=r.get("lab"), fluorescence=r.get("fluorescence"),
                        market=market)
        table.by_segment[seg] = res
        if res.depth is None:
            table.n_failed += 1
        else:
            table.n_resolved += 1
    log.info(table.summary())
    return table


# ---------------------------------------------------------------------------
# the pair, and the gap between them
# ---------------------------------------------------------------------------
# Thresholds for NAMING the gap. They are cut points on two 0-100 scores that
# are each already relative to this book, so they are presentation, not
# modelling — the underlying two numbers are always reported raw beside them.
_FAST = 60
_SLOW = 40
_DEEP = 60
_THIN = 40


def own_vs_market(own_velocity_score: float | None,
                  market_depth_score: int | None) -> dict:
    """The two numbers, their ratio, and the gap named in words.

    The ratio is >1 when this desk turns a segment over faster than the breadth
    of the market would suggest. That gap is the client's edge — MOU 5.2 calls
    it out specifically — so it is surfaced explicitly rather than left for a
    reader to divide two columns and notice.

    Returns them as three separate keys. There is deliberately no combined
    score: MOU 8.1 forbids it, and one blended "tradeability" number destroys
    the exact insight the client is paying for.
    """
    # NaN counts as missing, not as a number. This is not defensive padding: a
    # column of scores with some segments unresolved is stored by pandas as a
    # float column, and the unresolved entries arrive here as NaN rather than
    # None. Guarding only against None let NaN through to `round()`, which
    # raises — so the whole book failed the moment depth was genuinely partial,
    # which is the normal case.
    def _num(x):
        if x is None:
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return None if not math.isfinite(v) else v

    own, dep = _num(own_velocity_score), _num(market_depth_score)

    if own is None or dep is None:
        missing = "own velocity" if own is None else "market depth"
        return {"own_velocity_score": own, "market_depth_score": dep,
                "velocity_ratio": None,
                "edge": None,
                "edge_basis": f"not computable — {missing} unavailable"}

    ratio = round(own / dep, 2) if dep > 0 else None
    if own >= _FAST and dep <= _THIN:
        edge = "our edge — we turn it faster than the market's depth suggests"
        action = "keep stocking it; the thin market is not our problem"
    elif own <= _SLOW and dep >= _DEEP:
        edge = "market is deep but we do not turn it"
        action = "the segment trades — review price or assortment before blaming demand"
    elif own >= _FAST and dep >= _DEEP:
        edge = "liquid both ways"
        action = "healthy; watch margin rather than speed"
    elif own <= _SLOW and dep <= _THIN:
        edge = "genuinely illiquid — slow for us and thin in the market"
        action = "reprice to clear, or stop buying it"
    else:
        edge = "in line with the market"
        action = "no signal either way"

    return {
        "own_velocity_score": round(own),
        "market_depth_score": round(dep),
        "velocity_ratio": ratio,
        "edge": edge,
        "edge_basis": (f"own velocity {own:.0f}/100 vs market depth {dep:.0f}/100"
                       + (f" (ratio {ratio})" if ratio is not None else "")
                       + f" — {action}"),
    }
