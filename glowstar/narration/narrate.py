"""Generate the human-readable "why this price" explanation.

Two backends:
  * Claude API (when ANTHROPIC_API_KEY is set) — receives the computed facts as
    JSON and is asked for a bounded explanation. Its output is run through the
    number guard; if it invents a figure, we regenerate, and after a retry fall
    back to the deterministic template. The LLM never originates a number.
  * Deterministic template (default) — pure string assembly from the facts, so
    the system is fully runnable without any API key.

Latest Claude models (e.g. claude-opus-4-8) are the right default; the model id
is configurable via GS_LLM_MODEL.
"""

from __future__ import annotations

import json
import logging
import os

from .guard import validate

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a diamond pricing analyst. You are given a JSON object of "
    "ALREADY-COMPUTED pricing facts. Write a concise 2-3 sentence explanation "
    "of the suggested price for a trader. CRITICAL RULE: you may ONLY state "
    "numbers that appear in the JSON. Never invent, compute, or alter any "
    "figure. If a value is not in the JSON, do not mention it."
)


def template_narration(facts: dict) -> str:
    """Deterministic explanation assembled purely from computed facts."""
    d = facts["suggested_discount"]
    lo, hi = facts["ci_discount_low"], facts["ci_discount_high"]
    parts = [
        f"Suggested {d:.1f}% off Rap "
        f"(${facts['suggested_ppc']:,.0f}/ct, ${facts['suggested_net']:,.0f} net), "
        f"with an {int(facts.get('coverage_pct', 80))}% confidence range of "
        f"{lo:.1f}% to {hi:.1f}%."
    ]
    if facts.get("market_median_discount") is not None:
        parts.append(
            f"The market shows {facts['comparable_count']:,} comparable stones "
            f"clustering near {facts['market_median_discount']:.1f}% off Rap."
        )
    if facts.get("method") == "fallback":
        parts.append("This is a sparse-data fallback estimate — human review advised.")
    if facts.get("flags"):
        parts.append("Flags: " + ", ".join(facts["flags"]) + ".")
    return " ".join(parts)


def _claude_narration(facts: dict, model: str) -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=300, system=_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(facts, default=str)}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as e:  # network/auth/etc. — fall back, never crash pricing
        log.warning("Claude narration failed (%s); using template.", e)
        return None


def narrate(facts: dict) -> dict:
    """Return {'text', 'source', 'guard_ok'}. Number guard always enforced."""
    model = os.environ.get("GS_LLM_MODEL", "claude-opus-4-8")
    if os.environ.get("ANTHROPIC_API_KEY"):
        for _ in range(2):
            text = _claude_narration(facts, model)
            if text is None:
                break
            ok, bad = validate(text, facts)
            if ok:
                return {"text": text.strip(), "source": "claude", "guard_ok": True}
            log.warning("LLM emitted ungrounded numbers %s; regenerating.", bad)
    text = template_narration(facts)
    ok, _ = validate(text, facts)
    return {"text": text, "source": "template", "guard_ok": ok}
