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


def segment_keys(shape: str, weight: float, color: str, clarity: str) -> list[tuple]:
    """Most-specific -> least-specific segment keys for hierarchical backoff."""
    sb = size_band(weight)
    sh = (shape or "NA").strip().title()
    co = (color or "NA").strip().upper()
    cl = (clarity or "NA").strip().upper()
    return [
        (sh, sb, co, cl),
        (sh, sb, co),
        (sh, sb),
        (sh,),
        (),  # global
    ]
