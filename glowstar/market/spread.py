"""Spread (face-up size) premium bands — the client's diameter table.

WHAT THE CLIENT SENT
--------------------
    WEIGHT RANGE   DIAMETER     COL    CLA              CPS       FLO
    0.35-0.39      4.50-4.59    D E F  IF VVS1 VVS2     EX-EX-EX  NON FNT MED
    0.45-0.49      5.00-5.09    ...
    0.60-0.69      5.40-5.49    ...
    0.80-0.89      6.00-6.19    ...

WHY IT MATTERS
--------------
A round's weight does not fix how big it LOOKS. Weight hidden in the pavilion
makes a stone face up small; a well-spread stone of the same weight looks bigger
and trades at a premium. The diameter band is that "well-spread" window.

VERIFIED against the client's own 2,500+ rounds (live inventory, measured from
Length/Width): their actual diameters land right on these bands —
    0.35-0.39ct  median 4.56  (band 4.50-4.59)
    0.60-0.69ct  median 5.40  (band 5.40-5.49)
    0.45-0.49ct  median 4.93, p90 5.00  (band 5.00-5.09 = the top decile)
    0.80-0.89ct  median 5.91, p90 6.00  (band 6.00-6.19 = the top decile)
So the table is real and it selects the best-spread stones, not the average one.

THE RULE (as the desk stated it)
--------------------------------
  * A stone BELOW the band ("4.49 and under" for the 0.35-0.39 group) is NOT a
    premium stone — exclude it from the premium average.
  * A stone INSIDE the band (e.g. measuring 4.55 x 4.59) qualifies, and is
    averaged within its band.
Bands are treated as INCLUSIVE of both edges — the desk's own example counts
4.50 as in. Change `_INCLUSIVE` if they say otherwise.

This module only CLASSIFIES. It never prices: the premium a band earns is
measured from data (see `market/market_make.py`), never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# (weight_lo, weight_hi, diameter_lo, diameter_hi) — the client's table, verbatim.
PREMIUM_BANDS: tuple[tuple[float, float, float, float], ...] = (
    (0.35, 0.39, 4.50, 4.59),
    (0.45, 0.49, 5.00, 5.09),
    (0.60, 0.69, 5.40, 5.49),
    (0.80, 0.89, 6.00, 6.19),
)

# The premium table applies to ROUND stones: "diameter" is only well defined for a
# round outline. Fancies are judged on length x width ratio, a different rule.
PREMIUM_SHAPES = frozenset({"Round"})

_INCLUSIVE = True          # 4.50 counts as inside the 4.50-4.59 band


@dataclass(frozen=True)
class SpreadBand:
    weight_lo: float
    weight_hi: float
    diameter_lo: float
    diameter_hi: float

    @property
    def label(self) -> str:
        return f"{self.diameter_lo:.2f}-{self.diameter_hi:.2f}"

    @property
    def size_group(self) -> str:
        return f"{self.weight_lo:.2f}-{self.weight_hi:.2f}"


def diameter(length, width) -> float:
    """A round's face diameter = the mean of its two measured girdle diameters.

    A round is measured as length x width (e.g. 4.55 x 4.59); they differ slightly
    because no stone is perfectly circular. The trade quotes the average.
    """
    try:
        a, b = float(length), float(width)
    except (TypeError, ValueError):
        return float("nan")
    if not (a > 0 and b > 0):
        return float("nan")
    return (a + b) / 2.0


def band_for_weight(weight: float) -> SpreadBand | None:
    """The premium diameter band defined for this weight, if the table covers it."""
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return None
    for wlo, whi, dlo, dhi in PREMIUM_BANDS:
        if wlo <= w <= whi:
            return SpreadBand(wlo, whi, dlo, dhi)
    return None


def is_premium_spread(shape, weight, length, width) -> bool | None:
    """True/False if the table covers this stone, else None ("not applicable").

    None is NOT False. A 1.20ct round has no band in the client's table, so we
    cannot say it fails the premium test — saying so would quietly exclude it from
    an average it belongs in. Callers must treat None as "unknown", never as a fail.
    """
    if str(shape or "").strip().title() not in PREMIUM_SHAPES:
        return None
    band = band_for_weight(weight)
    if band is None:
        return None
    d = diameter(length, width)
    if not np.isfinite(d):
        return None                       # unmeasured stone: unknown, not a failure
    if _INCLUSIVE:
        return band.diameter_lo <= d <= band.diameter_hi
    return band.diameter_lo < d < band.diameter_hi


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add `diameter`, `spread_band`, `is_premium_spread` to a stone frame.

    Expects Shape_full / Weight / Length / Width. Missing measurement columns are
    tolerated (everything comes back unknown) so an external file without
    measurements still prices — it simply cannot claim a spread premium.
    """
    out = df.copy()
    if "Length" in out.columns and "Width" in out.columns:
        out["diameter"] = [diameter(l, w) for l, w in zip(out["Length"], out["Width"])]
    else:
        out["diameter"] = np.nan
    bands = [band_for_weight(w) for w in out.get("Weight", pd.Series(index=out.index))]
    out["spread_band"] = [b.label if b else None for b in bands]
    out["is_premium_spread"] = [
        is_premium_spread(s, w, l, wd)
        for s, w, l, wd in zip(out.get("Shape_full", pd.Series("", index=out.index)),
                               out.get("Weight", pd.Series(np.nan, index=out.index)),
                               out.get("Length", pd.Series(np.nan, index=out.index)),
                               out.get("Width", pd.Series(np.nan, index=out.index)))
    ]
    return out
