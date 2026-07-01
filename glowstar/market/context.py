"""Macro market-research context — the EXTERNAL, authentic market signal.

This is the "market research" the brief's north star leans on, distinct from
both the Uni cross-sectional anchor and the internal trend index: the broad,
sourced direction of the natural-diamond market (RAPI decline, lab-grown
substitution, tariffs, G7 traceability).

Design rules (authenticity first — the client's hard requirement):
  * Every figure carries a SOURCE and an as-of date (provenance). These are
    external priors, refreshable from authentic sources — not invented.
  * They are NEVER fed as silent numeric inputs to the price model. They are
    surfaced to the human, used to NARRATE, and used for a directional
    cross-check against the internal index (an authenticity guard). This keeps
    the macro view honest and auditable, and means a wrong/stale headline can
    never move a stone's price (no hallucination into the math).
  * NOT auto-scraped. News is curated and HUMAN-REVIEWED before it lands here —
    auto-ingesting unverified headlines would be the authenticity risk we are
    avoiding. The staleness guard below flags when a human refresh is due.

Refresh: a human updates these from the cited sources (rapaport.com price-index
releases, The Knot LGD report, GJEPC/White House tariff notices) and bumps
COMPILED_AS_OF. `current_context()` reports `is_stale` once that review is older
than STALE_AFTER_DAYS so an out-of-date macro view is surfaced, not trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

# When a human last reviewed/refreshed the signals below (bump on every refresh).
COMPILED_AS_OF = "2026-06"
# Flag the macro view stale if the last human review is older than this.
STALE_AFTER_DAYS = 60


@dataclass(frozen=True)
class Signal:
    metric: str
    value: str
    as_of: str
    direction: str        # "down" / "up" / "flat" for the market level
    source: str
    note: str = ""


# Seeded macro context (provenance-tagged). Refresh from source on ingest.
MACRO_SIGNALS: tuple[Signal, ...] = (
    Signal("RAPI 1ct (round)", "-30% since early 2024; -9.9% in 2025", "2026-01",
           "down", "rapaport.com price-index releases (FY2024/FY2025/Jan2026)",
           "Small goods hit hardest: 0.30ct -20%, 0.50ct -26% in 2025; 3ct+ roughly flat."),
    Signal("RAPI decline easing", "-1.3% in Jan 2026 (0.30ct/1ct)", "2026-02",
           "down", "rapaport.com 'Decline Eases' (Feb 4 2026)",
           "Rate of decline slowing; small goods possibly stabilizing into 2026."),
    Signal("Lab-grown share (US engagement)", "61% (2025), up from 52% (2024)", "2026-01",
           "down", "The Knot 2025/26 jewelry studies",
           "Substitution is the structural driver pulling natural prices down."),
    Signal("US tariffs on Indian goods",
           "50% (Aug 27 2025); Feb-2026 US-India framework to remove duties on natural "
           "cut/polished diamonds (jewelry ~18%), pending formalization", "2026-02",
           "flat", "White House US-India Joint Statement (Feb 2026); JCK; GJEPC",
           "Aug-2025 50% drove the H2 price shock; the Feb-2026 framework points to relief "
           "for loose natural diamonds — directional easing, not yet in force."),
    Signal("G7 origin due-diligence", "Statement required for natural polished >=0.5ct", "2026-01",
           "flat", "AWDC traceability update (Jan 1 2026)",
           "Paper-based, not a blockchain mandate; Tracr is voluntary. Compliance cost, not a price driver yet."),
)


def _staleness_days(compiled: str, today: date) -> int:
    """Days since the macro view was last human-reviewed (COMPILED_AS_OF)."""
    y, m = (int(p) for p in compiled.split("-")[:2])
    return (today - date(y, m, 1)).days


def current_context(today: date | None = None) -> dict:
    """Summary of the macro market state for surfacing in pricing output.

    Includes an explicit staleness flag: if the last human review is older than
    STALE_AFTER_DAYS, `is_stale` is True so the view is shown with a caveat
    rather than trusted blindly. (These signals never touch the price math.)
    """
    today = today or date.today()
    downs = sum(1 for s in MACRO_SIGNALS if s.direction == "down")
    ups = sum(1 for s in MACRO_SIGNALS if s.direction == "up")
    overall = "softening" if downs > ups else "firming" if ups > downs else "flat"
    stale_days = _staleness_days(COMPILED_AS_OF, today)
    is_stale = stale_days > STALE_AFTER_DAYS
    return {
        "overall_direction": overall,
        "headline": "Natural-diamond market in multi-year decline; rate of decline easing into 2026.",
        "signals": [asdict(s) for s in MACRO_SIGNALS],
        "as_of": max(s.as_of for s in MACRO_SIGNALS),
        "compiled_as_of": COMPILED_AS_OF,
        "staleness_days": stale_days,
        "is_stale": is_stale,
        "refresh_note": ("Macro priors are human-reviewed (not auto-scraped) and never enter "
                         "the price math. Refresh from the cited sources and bump COMPILED_AS_OF."
                         + (" REVIEW OVERDUE — surfaced with a staleness caveat." if is_stale else "")),
    }


def cross_check(internal_direction: str) -> dict:
    """Authenticity guard: does the client's own (internal index) direction agree
    with the external macro view? Disagreement is surfaced, not silently resolved.
    """
    macro = current_context()["overall_direction"]
    # Map internal index labels to macro vocabulary.
    internal = {"softening": "softening", "firming": "firming", "flat": "flat"}.get(
        internal_direction, "flat")
    agree = (macro == internal) or "flat" in (macro, internal)
    return {
        "internal_direction": internal,
        "macro_direction": macro,
        "agree": agree,
        "note": ("Internal trend and external macro view agree." if agree else
                 "DIVERGENCE: internal trend disagrees with the macro market view — "
                 "review before trusting the level (possible local/segment effect or data issue)."),
    }
