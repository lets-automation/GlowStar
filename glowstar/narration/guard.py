"""Number guard for LLM narration (brief Section 8, the hard rule).

The LLM receives already-computed numbers and may only narrate them. This guard
extracts every number from an LLM explanation and rejects the text if it
contains a figure not present in the structured facts (within rounding
tolerance). Numbers are computed, never hallucinated — enforced here.
"""

from __future__ import annotations

import re

# Matches integers/decimals with optional sign, %, $, and thousands separators.
_NUM_RE = re.compile(r"[-+]?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|[-+]?\$?\s?\d+(?:\.\d+)?%?")


def _to_float(token: str) -> float | None:
    t = token.replace("$", "").replace("%", "").replace(",", "").replace(" ", "")
    try:
        return float(t)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        v = _to_float(m)
        if v is not None:
            out.append(v)
    return out


def allowed_values(facts: dict) -> set[float]:
    """Every numeric value the LLM is permitted to mention.

    Includes the computed facts and a few harmless derived forms (absolute value
    of a discount, rounded integers) so natural phrasing like "53%" for a -53.0
    discount is accepted.
    """
    vals: set[float] = set()

    def add(v):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            f = float(v)
            vals.update({f, round(f, 1), round(f, 0), abs(f), round(abs(f), 1), round(abs(f), 0)})

    def walk(o):
        if isinstance(o, dict):
            for x in o.values():
                walk(x)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        else:
            add(o)

    walk(facts)
    return vals


def validate(text: str, facts: dict, *, tol: float = 0.5) -> tuple[bool, list[float]]:
    """Return (ok, offending_numbers).

    ok is False if the text contains any number not within `tol` of an allowed
    value. Small integers 0-12 are always allowed (counts, "4Cs", list items).
    """
    allowed = allowed_values(facts)
    offending = []
    for n in extract_numbers(text):
        if 0 <= n <= 12 and float(n).is_integer():
            continue
        if not any(abs(n - a) <= tol for a in allowed):
            offending.append(n)
    return (len(offending) == 0, offending)
