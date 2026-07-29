"""Average Market Make per size group (the client's 2026-07 spec).

THE RULE, AS THE DESK STATED IT
-------------------------------
  1. Group the stones by Size Group.
  2. Average EVERY criterion across the group: Diameter, Colour, Clarity, CPS, FLO.
  3. Match those AVERAGED values to the Market Data ranges.
  4. Filter market records satisfying ALL the averaged criteria together.
  5. Average the Market Make over those filtered records.

The order is load-bearing and the desk said so explicitly: *"The Market Data should
be selected only after the average values of all criteria have been calculated."*

Average-then-match is NOT the same as match-then-average. Pricing each stone
against its own comps and then averaging lets a handful of oddball stones drag the
benchmark; averaging the SPEC first asks one clean question — "what is the market
making for the typical stone in this group?" — which is the benchmark the desk
wants to quote against.

WHAT THE PIECES MEAN (confirmed with the client)
------------------------------------------------
  * "Market Data" = the live UNI feed (it carries colour/clarity/cut/fluorescence
    AND measurements, so all five averaged criteria are matchable).
  * "Market Make"  = the market's price level for that spec, i.e. the median
    discount off Rap — the same number the engine already uses everywhere else.
  * Size Group    = the client's own premium weight bands (market/spread.py), which
    is where the Diameter criterion comes from.

Averaging a GRADE: colour/clarity/cut/fluorescence are ordered categories, not
numbers. We average their RANK (D=0, E=1, F=2 ...) and round to the nearest grade,
which is what "average colour of D E F = E" means in the trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..reference.normalize import CLARITY_ORDER, normalize_fluorescence
from .spread import annotate, band_for_weight

log = logging.getLogger(__name__)

# Ordered vocabularies. Index = rank, so the mean of the ranks is the mean grade.
COLOR_SCALE: tuple[str, ...] = ("D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N")
CLARITY_SCALE: tuple[str, ...] = CLARITY_ORDER
CPS_SCALE: tuple[str, ...] = ("3EX", "EX", "VG", "GD", "FR")
# The GIA/trade fluorescence ladder has FIVE rungs. "Very Slight" and "Slight" are
# other labs' wording for the SAME rung as Faint, not extra steps between Faint and
# Medium — treating them as separate rungs stretches the scale and drags the average
# (None+Faint+Medium then averages to "Very Slight" instead of the correct "Faint").
FLUOR_SCALE: tuple[str, ...] = ("None", "Faint", "Medium", "Strong", "Very Strong")
_FLUOR_ALIAS: dict[str, str] = {"Very Slight": "Faint", "Slight": "Faint"}


def _fluor_rung(f) -> str:
    """Canonical fluorescence, with other labs' wording folded onto the trade rung."""
    c = normalize_fluorescence(f)
    return _FLUOR_ALIAS.get(c, c)


def _rank(value, scale: tuple[str, ...]) -> float:
    v = str(value or "").strip().upper()
    for i, s in enumerate(scale):
        if s.upper() == v:
            return float(i)
    return float("nan")


def _mean_grade(values, scale: tuple[str, ...]) -> str | None:
    """The average grade of a column: mean of ranks, rounded to the nearest grade."""
    ranks = np.array([_rank(v, scale) for v in values], dtype=float)
    ranks = ranks[np.isfinite(ranks)]
    if not len(ranks):
        return None
    return scale[int(round(float(np.mean(ranks))))]


def _cps_token(cps) -> str:
    """Collapse a CPS code to its leading cut grade ('EX-EX-EX' -> 'EX')."""
    s = str(cps or "").strip().upper().replace(" ", "")
    if s.startswith("3EX") or s in ("EX-EX-EX", "EXEXEX"):
        return "3EX"
    return s.split("-")[0] or "NA"


@dataclass
class GroupMake:
    """One size group's averaged spec and the market level that matches it."""
    size_group: str
    n_stones: int
    avg_diameter: float | None
    avg_color: str | None
    avg_clarity: str | None
    avg_cps: str | None
    avg_fluor: str | None
    market_make: float | None       # median market discount off Rap (negative)
    n_market: int = 0
    our_make: float | None = None   # our own stones' level, for comparison
    note: str = ""

    def as_row(self) -> dict:
        return {
            "Size group (ct)": self.size_group,
            "Stones": self.n_stones,
            "Avg diameter (mm)": None if self.avg_diameter is None else round(self.avg_diameter, 2),
            "Avg colour": self.avg_color, "Avg clarity": self.avg_clarity,
            "Avg cut": self.avg_cps, "Avg fluorescence": self.avg_fluor,
            "Market make (% below Rap)": None if self.market_make is None
                                         else round(abs(self.market_make), 1),
            "Market stones matched": self.n_market,
            "Our make (% below Rap)": None if self.our_make is None else round(abs(self.our_make), 1),
            "Note": self.note,
        }


def average_spec(group: pd.DataFrame) -> dict:
    """Step 2: the averaged criteria for one size group."""
    d = pd.to_numeric(group.get("diameter"), errors="coerce") if "diameter" in group else None
    return {
        "avg_diameter": float(d.mean()) if d is not None and d.notna().any() else None,
        "avg_color": _mean_grade(group.get("Color", []), COLOR_SCALE),
        "avg_clarity": _mean_grade(group.get("Clarity", []), CLARITY_SCALE),
        "avg_cps": _mean_grade([_cps_token(c) for c in group.get("CPS", [])], CPS_SCALE),
        "avg_fluor": _mean_grade([_fluor_rung(f) for f in group.get("Fluorescence", [])],
                                 FLUOR_SCALE),
    }


def _match_market(market: pd.DataFrame, spec: dict, band, diam_tol: float = 0.05) -> pd.DataFrame:
    """Step 3+4: market records satisfying ALL the averaged criteria together.

    Matching is on the averaged spec, not per stone. The diameter criterion uses the
    client's own premium band when the group has one (that is what the band is FOR);
    otherwise it falls back to a tolerance around the averaged diameter.
    """
    m = market
    if "color" in m:
        m = m[m["color"].astype(str).str.strip().str.upper() == str(spec["avg_color"]).upper()]
    if "clarity" in m:
        m = m[m["clarity"].astype(str).str.strip().str.upper() == str(spec["avg_clarity"]).upper()]
    if "cut" in m and spec["avg_cps"]:
        m = m[m["cut"].astype(str).map(_cps_token) == spec["avg_cps"]]
    if "fluorescence" in m and spec["avg_fluor"]:
        m = m[m["fluorescence"].map(_fluor_rung) == spec["avg_fluor"]]
    if "diameter" in m and spec["avg_diameter"] is not None:
        dm = pd.to_numeric(m["diameter"], errors="coerce")
        if band is not None:
            m = m[dm.between(band.diameter_lo, band.diameter_hi)]
        else:
            lo, hi = spec["avg_diameter"] - diam_tol, spec["avg_diameter"] + diam_tol
            m = m[dm.between(lo, hi)]
    return m


def average_market_make(stones: pd.DataFrame, market: pd.DataFrame,
                        min_market: int = 5) -> list[GroupMake]:
    """The full calculation, one row per size group.

    `stones` = the client's stones (needs Shape_full/Weight/Color/Clarity/CPS/
    Fluorescence, plus Length/Width for diameter).
    `market` = cleaned UNI records (color/clarity/cut/fluorescence/discount, and
    `diameter` when measurements are available).
    """
    s = annotate(stones)
    s = s[s["spread_band"].notna()]              # only groups the client's table defines
    out: list[GroupMake] = []
    for size_group, grp in s.groupby(s["Weight"].map(
            lambda w: (band_for_weight(w).size_group if band_for_weight(w) else None))):
        if size_group is None or grp.empty:
            continue
        spec = average_spec(grp)
        band = band_for_weight(float(grp["Weight"].iloc[0]))
        matched = _match_market(market, spec, band)
        disc = pd.to_numeric(matched.get("discount"), errors="coerce").dropna() \
            if len(matched) else pd.Series(dtype=float)
        ours = pd.to_numeric(grp.get("FDiscount"), errors="coerce").dropna() \
            if "FDiscount" in grp else pd.Series(dtype=float)
        enough = len(disc) >= min_market
        out.append(GroupMake(
            size_group=size_group, n_stones=len(grp), **spec,
            market_make=float(disc.median()) if enough else None,
            n_market=int(len(disc)),
            our_make=float(ours.median()) if len(ours) else None,
            note="" if enough else
                 f"only {len(disc)} market stones match this averaged spec — "
                 "too few for a reliable benchmark",
        ))
    return out


def to_frame(groups: list[GroupMake]) -> pd.DataFrame:
    return pd.DataFrame([g.as_row() for g in groups])
