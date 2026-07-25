"""Canonical normalization of diamond attributes.

The client's `records.json`, the Rapaport CSV grids, and the Uni market feed
each use slightly different spellings/codes for the same attributes. This
module maps them onto one canonical vocabulary and, critically, decides which
values the white Rapaport price list can and cannot price. Decisions here are
documented rules, never silent guesses (brief Sections 2, 5, 7.2).
"""

from __future__ import annotations

from enum import Enum

# --- Clarity ---------------------------------------------------------------

# Ordered best -> worst. This is the published Rapaport clarity axis plus the
# trade grade SI3 (present in the CSV grids) and FL (present in records.json,
# not separately published — Rapaport's top cell is IF).
CLARITY_ORDER: tuple[str, ...] = (
    "FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "SI3", "I1", "I2", "I3",
)
_CLARITY_SET = set(CLARITY_ORDER)

# Clarity grades the white Rap grids actually publish a cell for.
GRID_CLARITIES: tuple[str, ...] = (
    "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "SI3", "I1", "I2", "I3",
)


def normalize_clarity(raw: str) -> str:
    """Return a canonical clarity grade. Raises on an unknown value."""
    c = raw.strip().upper().replace(" ", "")
    if c in _CLARITY_SET:
        return c
    raise ValueError(f"Unknown clarity grade: {raw!r}")


def clarity_for_grid(clarity: str) -> str:
    """Map a canonical clarity to the grade used for Rap lookup.

    FL is priced at the IF cell (Rapaport publishes no separate FL cell).
    Documented rule, not a silent default.
    """
    c = normalize_clarity(clarity)
    return "IF" if c == "FL" else c


# --- Color -----------------------------------------------------------------

# The white Rapaport grids publish D..N. These are the only colors with a
# white-list price.
GRID_COLORS: tuple[str, ...] = ("D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N")
_GRID_COLOR_SET = set(GRID_COLORS)


def is_white_grid_color(raw: str) -> bool:
    """True iff `raw` is a single white color D..N that the grid can price.

    Fancy colors ("Fancy Vivid Yellow", "Faint Pink", ...), cape ranges below
    N ("O-P".."Y-Z"), and placeholders ("*") are NOT on the white list and must
    route to the attribute-based fallback (brief Sections 4.2, 7.6).
    """
    return raw.strip().upper() in _GRID_COLOR_SET


def normalize_color(raw: str) -> str:
    """Uppercase/trim a white color. Caller must check is_white_grid_color first."""
    return raw.strip().upper()


# --- Shape -----------------------------------------------------------------

class RapGrid(str, Enum):
    """Which published Rapaport list prices a shape."""

    ROUND = "round"
    FANCY = "fancy"  # the Pear list prices all fancy shapes (brief Section 4.2)


# Round is identified by the full name "Round" or the code "RBC"/"BR".
_ROUND_TOKENS = {"ROUND", "RBC", "BR", "RB", "RD", "BRILLIANT", "ROUND BRILLIANT"}


def grid_for_shape(shape_code: str | None, shape_full: str | None) -> RapGrid:
    """Return the Rap grid for a stone given its code and/or full name.

    Any shape that is not an unambiguous round prices off the fancy (Pear) list.
    """
    for token in (shape_full, shape_code):
        if token and token.strip().upper() in _ROUND_TOKENS:
            return RapGrid.ROUND
    return RapGrid.FANCY


# --- Fluorescence ----------------------------------------------------------

# records.json uses abbreviations; canonicalize to intensity buckets.
#
# MUST BE IDEMPOTENT: every canonical OUTPUT is also a key. It was not, and the gap
# was silent — "VERY STRONG" and "VERY SLIGHT" were absent, so
# normalize_fluorescence("Very Strong") returned "Unknown" while
# normalize_fluorescence("Vstg") returned "Very Strong". Any caller handed an
# already-normalised value (or the client's long-form spelling) would drop the
# stone into "Unknown" and quietly lose its fluorescence — on exactly the tier
# where fluorescence matters most.
FLUOR_CANON: dict[str, str] = {
    "NON": "None", "NONE": "None",
    "FNT": "Faint", "FAINT": "Faint",
    "VSL": "Very Slight", "VSLT": "Very Slight", "VERY SLIGHT": "Very Slight",
    "SLT": "Slight", "SLIGHT": "Slight",
    "MED": "Medium", "MEDIUM": "Medium",
    "STG": "Strong", "STRONG": "Strong",
    "VSTG": "Very Strong", "VSTRONG": "Very Strong", "VERY STRONG": "Very Strong",
    # Uni feed single-letter intensities:
    "N": "None", "F": "Faint", "M": "Medium", "S": "Strong", "VS": "Very Strong",
}


def normalize_fluorescence(raw: str | None) -> str:
    """Canonical fluorescence intensity, or 'Unknown' if missing/unmapped.

    Idempotent: normalize(normalize(x)) == normalize(x) — see FLUOR_CANON.
    """
    if raw is None:
        return "Unknown"
    return FLUOR_CANON.get(str(raw).strip().upper(), "Unknown")
