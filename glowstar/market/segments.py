"""Shared market-segment helpers: size bands and segment keys.

Kept tiny and dependency-free so both the streaming aggregator and the live
anchor use identical segment definitions.
"""

from __future__ import annotations

import bisect

# Size-band lower edges (Rapaport-style brackets, with the 6-9.99 gap and 10+
# as their own bands). A weight maps to the index of the band it falls in.
SIZE_EDGES: tuple[float, ...] = (
    0.0, 0.18, 0.23, 0.30, 0.40, 0.50, 0.70, 0.90, 1.00, 1.50,
    2.00, 3.00, 4.00, 5.00, 6.00, 10.00, 11.00,
)


def size_band(weight: float) -> int:
    """Index of the size band containing `weight` (0-based)."""
    return max(0, bisect.bisect_right(SIZE_EDGES, weight) - 1)


def size_tag(weight: float) -> str:
    """A 0.10ct sub-bucket tag within a Rap bracket (e.g. 0.80 -> '#0.80').

    Rap $/ct is flat across a bracket, but DEMAND is not — within 0.70-0.89,
    0.80ct (near the 0.90 break) lists materially shallower than 0.70ct. A single
    bracket-level market median lumps these and over-discounts the upper part. The
    LIVE market is keyed by this finer sub-bucket so a 0.80 stone anchors to
    0.80-0.89 comps, not the deeper 0.70-0.79 ones. Banked (bracket-level) stays
    the fallback for thin sub-buckets.
    """
    import math
    return "#%.2f" % (math.floor(float(weight) / 0.10) * 0.10)


def size_bucket_window(weight: float) -> tuple[float, float]:
    """The [lo, hi] 0.10ct sub-bucket for `weight`, clamped to its Rap bracket."""
    import math
    b = size_band(weight)
    blo = SIZE_EDGES[b]
    bhi = round(SIZE_EDGES[b + 1] - 0.01, 2) if b + 1 < len(SIZE_EDGES) else round(weight + 1.0, 2)
    lo = max(blo, round(math.floor(float(weight) / 0.10) * 0.10, 2))
    hi = min(bhi, round(lo + 0.0999, 2))
    return lo, hi


# The coarse cut tiers a CPS code maps to (see cut_tier). A segment key is
# "cut-aware" iff its LAST token is one of these — used to back off along the
# cut-matched hierarchy (5->4->3->2 tuples) without ever falling to a cut-BLIND
# level (whose last token is a colour/clarity/size, never a tier).
CUT_TIERS: tuple[str, ...] = ("EX", "VG", "LOW")


def is_cut_aware_key(key: tuple) -> bool:
    """True if `key` is a cut-matched segment (its last token is a cut tier)."""
    return bool(key) and key[-1] in CUT_TIERS


# GIA grades overall CUT only for round brilliants. Fancy shapes have polish &
# symmetry but NO cut grade, so cut-tier matching applies to rounds only — a
# fancy stone prices to the 4C market, not a (mismatched) cut tier.
_CUT_GRADED_SHAPES = {"Round"}


def cut_graded(shape) -> bool:
    return str(shape or "").strip().title() in _CUT_GRADED_SHAPES


def cut_tier(cps) -> str:
    """Coarse cut tier that materially moves price: EX (incl. 3EX) / VG / LOW.

    Measured on the live market: for the SAME 4Cs, VG-cut rounds trade ~10
    points deeper (cheaper) than EX/3EX. Anchoring a VG stone to an EX-dominated
    market over-prices it — so the market segment must include the cut tier.
    The cut grade is the leading token of the CPS code (e.g. '3EX'->EX,
    'VG-EX'->VG). Unknown -> EX (don't over-discount an unknown).
    """
    first = str(cps or "").upper().strip().split("-")[0].lstrip("3").strip()
    if first in ("EX", "ID", "EXCELLENT", "IDEAL"):
        return "EX"
    if first in ("VG", "VERY GOOD"):
        return "VG"
    if not first:
        return "EX"
    return "LOW"      # GD / FR / PR / etc.


def segment_keys(shape: str, weight: float, color: str, clarity: str,
                 cut=None) -> list[tuple]:
    """Most-specific -> least-specific segment keys for hierarchical backoff.

    When `cut` is supplied, a CUT-AWARE most-specific level is added at the front
    so a stone is matched to its own cut tier first, then backs off to cut-blind
    levels if there isn't enough cut-matched support.
    """
    sb = size_band(weight)
    sh = (shape or "NA").strip().title()
    co = (color or "NA").strip().upper()
    cl = (clarity or "NA").strip().upper()
    base = [
        (sh, sb, co, cl),
        (sh, sb, co),
        (sh, sb),
        (sh,),
        (),  # global
    ]
    if cut is not None:
        # CUT-AWARE backoff at several granularities BEFORE any cut-blind level,
        # so a VG stone always matches a VG market (never the EX-dominated blend),
        # even when the most-specific cut+4C segment is thin.
        ct = cut_tier(cut)
        return [(sh, sb, co, cl, ct), (sh, sb, co, ct), (sh, sb, ct), (sh, ct)] + base
    return base
