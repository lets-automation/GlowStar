"""Macro market-research context — the EXTERNAL, authentic market signal.

This is the "market research" the brief's north star leans on, distinct from
both the Uni cross-sectional anchor and the internal trend index: the broad,
sourced direction of the natural-diamond market (RAPI decline, lab-grown
substitution, tariffs, G7 traceability).

Design rules:
  * Every figure carries a SOURCE and an as-of date (provenance). These are
    external priors, refreshable from authentic sources — not invented.
  * They are NEVER fed as silent numeric inputs to the price model. They are
    surfaced to the human, used to NARRATE, and used for a directional
    cross-check against the internal index (an authenticity guard). This keeps
    the macro view honest and auditable.

Refresh: replace the seeded values from the live source on each ingest cycle
(rapaport.com price-index releases, The Knot LGD report, GJEPC tariff notices).
The seed below was compiled 2026-06 from the cited sources.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


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
    Signal("US tariffs on Indian goods", "25% (Aug 7 2025) -> 50% (Aug 27 2025)", "2025-08",
           "down", "GJEPC / JCK Feb 2026",
           "Triggered the H2-2025 price shock; natural-diamond relief prospective/unsigned."),
    Signal("G7 origin due-diligence", "Statement required for natural polished >=0.5ct", "2026-01",
           "flat", "AWDC traceability update (Jan 1 2026)",
           "Paper-based, not a blockchain mandate; Tracr is voluntary. Compliance cost, not a price driver yet."),
)


def current_context() -> dict:
    """Summary of the macro market state for surfacing in pricing output."""
    downs = sum(1 for s in MACRO_SIGNALS if s.direction == "down")
    ups = sum(1 for s in MACRO_SIGNALS if s.direction == "up")
    overall = "softening" if downs > ups else "firming" if ups > downs else "flat"
    return {
        "overall_direction": overall,
        "headline": "Natural-diamond market in multi-year decline; rate of decline easing into 2026.",
        "signals": [asdict(s) for s in MACRO_SIGNALS],
        "as_of": max(s.as_of for s in MACRO_SIGNALS),
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
