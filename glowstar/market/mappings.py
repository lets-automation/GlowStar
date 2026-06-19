"""Uni Diamonds request codebook (brief Section 3.3 — the blocking CONFIRM item).

The Uni feed RESPONSES are already in plain strings (shape:"Round",
color:"D", ...), so parsing needs no codebook. The codebook is only needed to
BUILD request filters, where values are numeric codes.

Only the mappings actually proven by the documented request<->response pairing
are marked CONFIRMED. Everything else is INFERRED (a plausible sequential guess)
and MUST be verified with the client/Uni before being used to drive live
queries. `to_code(..., strict=True)` raises on any non-confirmed value so we
fail loudly rather than silently querying the wrong stones.

Proven by API-Documentation.docx (request -> response-uni.json):
  shape:1 -> Round | color:1 -> D | clarity:1 -> IF, clarity:2 -> VVS1
  lab_ids:1 -> GIA | fluorescence_intensity:7 -> Faint | country_id:99 -> India
"""

from __future__ import annotations

# value -> numeric code. CONFIRMED entries are verified; INFERRED are guesses.

# NOTE: the API doc example implied clarity IF=1; LIVE calibration disproved it
# (code 1 = FL, 2 = IF, 3 = VVS1). The verified values below are from the live
# API, not the doc. This is why we calibrate instead of trusting the doc.
CONFIRMED: dict[str, dict[str, int]] = {
    "shape": {"Round": 1},
    "color": {"D": 1},
    "clarity": {"FL": 1, "IF": 2, "VVS1": 3},
    "lab": {"GIA": 1},
    "country": {"India": 99},
}

# Plausible sequential extensions — NOT verified. Do not use for live queries
# until confirmed; kept here only to document the hypothesis to check.
INFERRED: dict[str, dict[str, int]] = {
    "color": {"E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7, "K": 8, "L": 9, "M": 10, "N": 11},
    "clarity": {"VVS2": 3, "VS1": 4, "VS2": 5, "SI1": 6, "SI2": 7, "SI3": 8, "I1": 9, "I2": 10, "I3": 11},
}


class UnmappedCodeError(ValueError):
    """Raised when a value has no confirmed Uni code and strict mode is on."""


def _load_verified() -> dict[str, dict[str, int]]:
    """Load the codebook empirically verified against the live Uni API, if built
    (artifacts/uni_codebook.json via market.calibrate_codebook). These are facts
    confirmed from real API responses and are treated as CONFIRMED."""
    try:
        from ..config import ARTIFACTS_DIR
        path = ARTIFACTS_DIR / "uni_codebook.json"
        if path.exists():
            import json
            return {f: {k: int(v) for k, v in m.items()}
                    for f, m in json.loads(path.read_text(encoding="utf-8")).items()}
    except Exception:
        pass
    return {}


# Verified-from-live codes override/extend the doc-confirmed seed.
VERIFIED: dict[str, dict[str, int]] = _load_verified()
for _field, _m in VERIFIED.items():
    CONFIRMED.setdefault(_field, {}).update(_m)


def to_code(field: str, value: str, *, strict: bool = True) -> int:
    """Return the Uni numeric code for a field value.

    strict=True (default): only CONFIRMED/VERIFIED mappings are returned;
    anything else raises UnmappedCodeError. strict=False also allows INFERRED
    guesses (experimentation only) and still raises if entirely unknown.
    """
    confirmed = CONFIRMED.get(field, {})
    if value in confirmed:
        return confirmed[value]
    if not strict and value in INFERRED.get(field, {}):
        return INFERRED[field][value]
    raise UnmappedCodeError(
        f"No {'confirmed ' if strict else ''}Uni code for {field}={value!r}. "
        "Run market.calibrate_codebook or confirm the mapping before querying live."
    )


def is_confirmed(field: str, value: str) -> bool:
    return value in CONFIRMED.get(field, {})
