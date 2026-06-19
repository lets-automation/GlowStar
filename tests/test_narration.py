"""Tests for the LLM number guard (brief Section 8) and template narration.

The guard is a correctness control: it must reject any explanation containing a
number that was not computed, while accepting natural phrasings of the allowed
numbers.
"""

from __future__ import annotations

from glowstar.narration.guard import validate, extract_numbers
from glowstar.narration.narrate import template_narration, narrate

FACTS = {
    "suggested_discount": -53.0,
    "suggested_ppc": 4700.0,
    "suggested_net": 5640.0,
    "ci_discount_low": -58.0,
    "ci_discount_high": -48.0,
    "comparable_count": 1200,
    "market_median_discount": -52.0,
    "method": "model+anchor",
    "flags": [],
    "coverage_pct": 80,
}


def test_guard_accepts_grounded_text():
    text = "Suggested -53% off Rap, range -58% to -48%; market clusters near -52%."
    ok, bad = validate(text, FACTS)
    assert ok and bad == []


def test_guard_rejects_invented_number():
    text = "Suggested -53% off Rap, but I'd really price this at -41% to be safe."
    ok, bad = validate(text, FACTS)
    assert not ok
    assert 41.0 in [abs(b) for b in bad]


def test_template_narration_is_self_consistent():
    text = template_narration(FACTS)
    ok, bad = validate(text, FACTS)
    assert ok, f"template emitted ungrounded numbers: {bad}"


def test_narrate_without_api_key_uses_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = narrate(FACTS)
    assert out["source"] == "template"
    assert out["guard_ok"] is True


def test_small_counts_allowed():
    # "4Cs" and list-like small integers must not trip the guard.
    ok, _ = validate("The 4Cs drive this; 3 factors matter.", FACTS)
    assert ok
