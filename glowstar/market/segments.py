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


# The client's ROUND price slots (their Master-grid cell boundaries). These are
# the real, IRREGULAR trade slots — e.g. within 0.30-0.39 the splits are .32/.35
# but within 0.80-0.89 they are .83/.85 — so a formula won't do; they are listed
# exactly. A 0.84ct and a 0.85ct round therefore price in DIFFERENT slots, not one
# lumped 0.80-0.89 bucket. (Client instruction, applied to ROUND pricing.)
ROUND_SLOTS: tuple[tuple[float, float], ...] = (
    (0.30, 0.31), (0.32, 0.34), (0.35, 0.39),
    (0.40, 0.41), (0.42, 0.44), (0.45, 0.49),
    (0.50, 0.51), (0.52, 0.53), (0.54, 0.59),
    (0.60, 0.62), (0.63, 0.64), (0.65, 0.69),
    (0.70, 0.72), (0.73, 0.74), (0.75, 0.79),
    (0.80, 0.82), (0.83, 0.84), (0.85, 0.89),
    (0.90, 0.92), (0.93, 0.94), (0.95, 0.99),
)
_ROUND_SLOT_LOS = [s[0] for s in ROUND_SLOTS]


def round_slot(weight: float) -> tuple[float, float] | None:
    """The client's round slot [lo, hi] containing `weight`, or None if outside
    the specified 0.30-0.99 range (falls back to the 0.10ct bucket there)."""
    w = float(weight)
    i = bisect.bisect_right(_ROUND_SLOT_LOS, w) - 1
    if 0 <= i < len(ROUND_SLOTS):
        lo, hi = ROUND_SLOTS[i]
        if lo <= w <= hi:
            return (lo, hi)
    return None


def size_bucket_window(weight: float, shape=None) -> tuple[float, float]:
    """The size sub-slot [lo, hi] for a stone. For ROUND stones this is the
    client's exact price slot (e.g. 0.84 -> 0.83-0.84, distinct from 0.85-0.89);
    otherwise a 0.10ct sub-bucket clamped to the Rap bracket."""
    import math
    if cut_graded(shape):                       # rounds -> client slots
        slot = round_slot(weight)
        if slot is not None:
            return slot
    b = size_band(weight)
    blo = SIZE_EDGES[b]
    bhi = round(SIZE_EDGES[b + 1] - 0.01, 2) if b + 1 < len(SIZE_EDGES) else round(weight + 1.0, 2)
    lo = max(blo, round(math.floor(float(weight) / 0.10) * 0.10, 2))
    hi = min(bhi, round(lo + 0.0999, 2))
    return lo, hi


def size_tag(weight: float, shape=None) -> str:
    """Sub-slot tag keying the LIVE market segment. For ROUND stones this is the
    client's price slot (so 0.84 and 0.85 key to DIFFERENT market segments);
    otherwise a 0.10ct sub-bucket. Banked (bracket-level) is the thin-slot fallback.
    """
    import math
    if cut_graded(shape):
        slot = round_slot(weight)
        if slot is not None:
            return "#%.2f-%.2f" % slot
    return "#%.2f" % (math.floor(float(weight) / 0.10) * 0.10)


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
