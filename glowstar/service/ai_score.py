"""AI Score — the 0-100 scores their FrontOffice response carries.

CLIENT DECISION (2026-07-30): **Demand Score is removed.** It required per-stone
enquiry / offer / video-view data, and `GetCustomerSearchHistories` does not carry
it: every row is `searchType: "Inventory"` with `customer` and `sellerId` empty,
so we could see WHAT was searched but never WHO wanted it or how badly. Rather
than dress a weak proxy up as demand, the score is built from what is actually
measurable.

Removing it forced two of the original eight to be re-based, because they were
defined ON demand:

  * MARKET STRENGTH was "use the Demand Score". It is now segment health measured
    from the market itself — how much depth the segment carries and how steadily
    this desk sells into it.
  * URGENCY was "inventory age AND current demand". It is now age measured
    against that segment's own expected days-to-sell, which is the part that was
    ever actionable: a 120-day-old stone in a 40-day segment is late regardless
    of what demand is doing.

THE SIX COMPONENTS
------------------
  Competition        how crowded the segment is (more rivals = harder = lower)
  Liquidity          how reliably this desk turns that segment over
  Price Competitive  where our price sits against the market level
  Turnaround         expected days to sell (from the censoring-corrected table)
  Market Strength    depth + our own sales momentum in the segment
  Urgency            how overdue this specific stone is vs its segment
  Confidence         how much data stands behind all of the above

CONVENTION: 100 is always GOOD FOR THE CLIENT. High Competition = little
competition. High Urgency = NOT urgent (comfortable). A single direction means a
low number always reads as "look at this", which is the point of a score.

ON THE WEIGHTS — read before changing them
------------------------------------------
The weights below are NOT fitted. There is no outcome label to fit them against
yet: "was this a good price?" is only answerable once a stone sells, and the
scores are meant to be used BEFORE that. So they are a documented, deliberately
flat starting point, not a discovered optimum, and `FinalAIScore` ships with the
component breakdown beside it so the desk can see what drove it and disagree.

To make them real: once enough scored stones have sold, regress the components
against an outcome the client actually cares about (sold within N days? at what
margin?) and refit. That is a measured change with a before/after — not a knob
to twist because a number looks low.
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)

# Flat by design — see the module docstring. They must sum to 1.0.
WEIGHTS: dict[str, float] = {
    "Competition": 0.15,
    "Liquidity": 0.20,
    "PriceCompetitive": 0.20,
    "Turnaround": 0.15,
    "MarketStrength": 0.15,
    "Urgency": 0.15,
}


def _clamp(x: float) -> int:
    return int(max(0, min(100, round(x))))


def competition_score(market_depth: int | None) -> tuple[int | None, str]:
    """How crowded is this stone's segment? Fewer rivals scores HIGHER.

    `market_depth` = comparable live listings the market carries. Log-scaled: the
    difference between 5 and 50 rivals matters far more than 5,000 vs 5,050.
    """
    if market_depth is None:
        return None, "no market data for this segment"
    n = max(0, int(market_depth))
    if n == 0:
        return 100, "no comparable listings — you are effectively alone"
    score = 100.0 - 22.0 * math.log10(max(1.0, n))       # 10 -> 78, 1k -> 34, 100k -> -10
    return _clamp(score), f"{n:,} comparable listings in the market"


def liquidity_score(own_sales: int | None, median_days: float | None) -> tuple[int | None, str]:
    """How reliably does THIS desk turn this segment over?

    Two things together: how often they sell it, and how long it takes. A segment
    they sell often AND quickly is liquid for them regardless of the broad market
    — which is the client's own stated nuance (their edge lives exactly there).
    """
    if own_sales is None and median_days is None:
        return None, "no sales history for this segment"
    parts, why = [], []
    if own_sales is not None:
        parts.append(min(100.0, 20.0 * math.log10(max(1, own_sales) + 1) * 1.6))
        why.append(f"{own_sales} own sales")
    if median_days is not None:
        parts.append(max(0.0, 100.0 - (median_days / 180.0) * 100.0))
        why.append(f"typically {median_days:.0f}d to sell")
    return _clamp(sum(parts) / len(parts)), "; ".join(why)


def price_competitive_score(our_discount: float | None,
                            market_discount: float | None) -> tuple[int | None, str]:
    """Is our asking price attractive against the market level?

    Deeper than market (cheaper) scores high; shallower (dearer) scores low. This
    is the client's own framing: if the market sells at -30 they cannot ask -25,
    and if the market sells at -40 they should not give -50.
    """
    if our_discount is None or market_discount is None:
        return None, "no market level to compare against"
    gap = our_discount - market_discount     # negative = we are deeper = cheaper
    score = 50.0 - gap * 6.0                 # 2 pts cheaper -> 62; 2 pts dearer -> 38
    if gap < -12:
        score = min(score, 70.0)             # far under market is a margin leak, not a win
    where = ("cheaper than" if gap < -0.5 else
             "dearer than" if gap > 0.5 else "in line with")
    return _clamp(score), f"{abs(gap):.1f} pts {where} the market level"


def turnaround_score(median_days: float | None) -> tuple[int | None, str]:
    """Expected days to sell. Faster scores higher."""
    if median_days is None:
        return None, "no days-to-sell estimate for this segment"
    return _clamp(100.0 - (median_days / 180.0) * 100.0), \
        f"expected ~{median_days:.0f} days to sell"


def market_strength_score(market_depth: int | None, own_sales: int | None,
                          median_days: float | None) -> tuple[int | None, str]:
    """Overall health of the segment's market.

    Re-based after Demand was removed: a healthy segment is one that carries real
    market depth AND that this desk sells into steadily. Depth alone is not
    strength (a flooded segment is deep and weak), so it is combined with the
    desk's own turnover rather than used on its own.
    """
    if market_depth is None and own_sales is None:
        return None, "not enough market or sales data"
    parts, why = [], []
    if market_depth is not None:
        d = min(100.0, 18.0 * math.log10(max(1, market_depth) + 1))   # a live segment
        parts.append(d)
        why.append(f"{market_depth:,} live listings")
    if own_sales is not None and median_days is not None:
        parts.append(max(0.0, 100.0 - (median_days / 180.0) * 100.0))
        why.append(f"desk turns it in ~{median_days:.0f}d")
    return _clamp(sum(parts) / len(parts)), "; ".join(why)


def urgency_score(age_days: float | None, expected_days: float | None) -> tuple[int | None, str]:
    """How overdue is THIS stone? 100 = comfortable, low = act now.

    Age is judged against the stone's OWN segment, not a flat number of days: 90
    days is unremarkable in a segment that takes 120 and alarming in one that
    takes 30.
    """
    if age_days is None:
        return None, "stone age unknown"
    if not expected_days or expected_days <= 0:
        expected_days = 60.0
    ratio = age_days / expected_days
    score = 100.0 - (ratio - 1.0) * 70.0 if ratio > 1 else 100.0 - (ratio * 10.0)
    if ratio <= 1:
        note = f"{age_days:.0f}d old, within the ~{expected_days:.0f}d norm"
    else:
        note = (f"{age_days:.0f}d old vs a ~{expected_days:.0f}d norm "
                f"({ratio:.1f}x) — consider repricing")
    return _clamp(score), note


def final_score(components: dict[str, int | None]) -> tuple[int | None, str]:
    """Weighted blend of whatever components could actually be computed.

    Missing components are dropped and the remaining weights renormalised, so a
    stone with thin data gets an honest score from what IS known rather than a
    penalty for data it was never going to have. If nothing is computable the
    answer is None — never a default number dressed up as an assessment.
    """
    usable = {k: v for k, v in components.items() if v is not None and k in WEIGHTS}
    if not usable:
        return None, "no component could be computed"
    total_w = sum(WEIGHTS[k] for k in usable)
    score = sum(WEIGHTS[k] * v for k, v in usable.items()) / total_w
    missing = [k for k in WEIGHTS if k not in usable]
    note = f"from {len(usable)}/{len(WEIGHTS)} components"
    if missing:
        note += f" (missing: {', '.join(missing)})"
    return _clamp(score), note


def compute(*, our_discount: float | None, market_discount: float | None,
            market_depth: int | None, own_sales: int | None,
            median_days: float | None, age_days: float | None,
            confidence: int | None) -> dict:
    """All scores for one stone, each with the reason behind it.

    Every score ships with its `why` — a bare number the desk cannot interrogate
    is a number they will (rightly) stop trusting.
    """
    comp, comp_w = competition_score(market_depth)
    liq, liq_w = liquidity_score(own_sales, median_days)
    pc, pc_w = price_competitive_score(our_discount, market_discount)
    ta, ta_w = turnaround_score(median_days)
    ms, ms_w = market_strength_score(market_depth, own_sales, median_days)
    urg, urg_w = urgency_score(age_days, median_days)

    parts = {"Competition": comp, "Liquidity": liq, "PriceCompetitive": pc,
             "Turnaround": ta, "MarketStrength": ms, "Urgency": urg}
    final, final_w = final_score(parts)
    return {
        "CompetitionScore": comp,
        "LiquidityScore": liq,
        "PriceCompetitiveScore": pc,
        "TurnaroundScore": ta,
        "MarketStrengthScore": ms,
        "UrgencyScore": urg,
        "ConfidenceScore": confidence,
        "FinalAIScore": final,
        "ScoreBasis": {
            "Competition": comp_w, "Liquidity": liq_w, "PriceCompetitive": pc_w,
            "Turnaround": ta_w, "MarketStrength": ms_w, "Urgency": urg_w,
            "Final": final_w,
        },
        "DemandScore": None,
        "DemandScoreStatus": "removed at client request (2026-07-30) — the search "
                             "feed carries no buyer identity, offers or video views",
    }
