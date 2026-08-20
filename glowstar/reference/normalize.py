"""Canonical normalization of diamond attributes.

The client's `records.json`, the Rapaport CSV grids, and the Uni market feed
each use slightly different spellings/codes for the same attributes. This
module maps them onto one canonical vocabulary and, critically, decides which
values the white Rapaport price list can and cannot price. Decisions here are
documented rules, never silent guesses (brief Sections 2, 5, 7.2).
"""

from __future__ import annotations

import re
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


# Trade shape codes / spellings -> the exact `Shape_full` the engine trains on.
#
# MUST BE IDEMPOTENT, exactly like FLUOR_CANON below: every canonical OUTPUT is
# also a key, so passing an already-canonical value through is a no-op.
#
# WHY THIS LIVES HERE AND NOT IN reporting/
# ----------------------------------------
# This map existed only in `reporting/price_file.py`, so the EXCEL path
# canonicalised shapes and the API path did not. The engine routes on a raw dict
# lookup (`_shape_counts.get(row["Shape_full"])`), and the client's own inventory
# API sends CODES — `RBC`, `OB`, `PB`, `MB` — never `"Round"`. So a stone the
# desk calls RBC scored 0 training rows, was flagged `rare_shape`, and dropped to
# the sparse-data fallback: measured -57.58 instead of -51.44 on a 1.01 G VS1,
# **6.1 points deeper**, on the single most common stone there is.
#
# The Excel files we have been sending were fine. The endpoint the CRM is about
# to call was not. Canonicalise at every entry point, from one table.
_SHAPE_FULL: dict[str, str] = {
    "ROUND": "Round", "RBC": "Round", "RB": "Round", "BR": "Round", "RD": "Round",
    "BRILLIANT": "Round", "ROUND BRILLIANT": "Round",
    # NOTE: `S.OV` is deliberately NOT mapped here. It is a DISTINCT trained
    # level (24 stones) meaning a stepped oval, not a synonym for Oval (3,654).
    # Mapping it collapsed a real category the model prices separately.
    "OVAL": "Oval", "OB": "Oval", "OV": "Oval", "F.OVAL": "Oval",
    "PEAR": "Pear", "PB": "Pear", "PS": "Pear", "PMB": "Pear",
    "MARQUISE": "Marquise", "MB": "Marquise", "MQ": "Marquise",
    "HEART": "Heart", "HB": "Heart", "HS": "Heart",
    "EMERALD": "Emerald", "EM": "Emerald", "EB": "Emerald", "EC": "Emerald",
    "PRINCESS": "Princess", "PR": "Princess", "SMB": "Princess",
    "CUSHION": "Cushion", "CB": "Cushion", "CU": "Cushion", "CMB": "Cushion",
    # RADIANT -> "Cut-Cornered Rectangular", NOT "Radiant".
    # "Radiant" has ZERO training rows; the name the client's inventory uses, and
    # therefore the name the model learned, is "Cut-Cornered Rectangular" (183
    # rows). Pointing these at "Radiant" sent every radiant to an unseen category:
    # the same stone priced -57.82 under the inventory name and -54.50 under the
    # trade code, a 3.3-point spread decided purely by which spelling arrived.
    "RADIANT": "Cut-Cornered Rectangular", "CCRMB": "Cut-Cornered Rectangular",
    "RA": "Cut-Cornered Rectangular", "CCSMB": "Cut-Cornered Rectangular",
    "RMB": "Cut-Cornered Rectangular",
    "CUT-CORNERED RECTANGULAR": "Cut-Cornered Rectangular",
    "CUT CORNERED RECTANGULAR": "Cut-Cornered Rectangular",
    # Cushion variants are DISTINCT trained levels (Cushion Long 43, Cushion
    # Brilliant 19, Cushion 15) — uppercase input must land on the exact spelling.
    "CUSHION LONG": "Cushion Long", "CUSHION BRILLIANT": "Cushion Brilliant",
    "CUSHION MODIFIED BRILLIANT": "Cushion Brilliant",
    "SQ. EMERALD": "Sq. Emerald", "SQ.EMERALD": "Sq. Emerald",
    "SQ EMERALD": "Sq. Emerald", "SQEM": "Sq. Emerald", "SEM": "Sq. Emerald",
    "SE": "Sq. Emerald",
}


# --- Cut / Polish / Symmetry (CPS) -----------------------------------------

# The model learned the cut effect from a SMALL closed vocabulary: 3EX, EX, VG,
# GD, FR, PR (see reporting/price_file._make_cps, which produces exactly these).
# Any other spelling is a category it has never seen.
#
# The client has already caught a CPS-vocabulary bug once. This is the same bug
# on the new API path: `/price` accepts CPS as a free string, so three spellings
# of the SAME triple-excellent stone returned three different prices —
#   "3EX"      -> -45.85   (the trained form: correct)
#   "EX-EX-EX" -> -51.44
#   "EX EX EX" -> -59.54   (identical to sending NO cut data at all)
# a 13.7-point spread on one stone, decided purely by punctuation.
_CPS_GRADE: dict[str, str] = {
    "EXCELLENT": "EX", "EX": "EX", "IDEAL": "EX", "ID": "EX", "X": "EX",
    "VERY GOOD": "VG", "VERYGOOD": "VG", "VG": "VG", "V.GOOD": "VG",
    "GOOD": "GD", "GD": "GD", "G": "GD",
    "FAIR": "FR", "FR": "FR", "F": "FR",
    # POOR -> "FR", not "PR". The model has never seen "PR": the trained CPS
    # vocabulary is {3EX, EX, VG, GD, FR, VG-GD}. Emitting "PR" produced an
    # unseen category that HistGradientBoosting drops silently, so a Poor-cut
    # stone lost its cut signal entirely. FR is the worst grade the model
    # actually knows, which is the closest honest answer. Found by the
    # bidirectional guard, not by anyone noticing.
    "POOR": "FR", "PR": "FR", "P": "FR",
}

# Whole-string spellings of triple-excellent that carry no separators to split on.
_TRIPLE_EX = {"3EX", "3X", "XXX", "EXEXEX", "TRIPLEEX", "TRIPLEEXCELLENT", "3EXCELLENT"}

# The COMPLETE vocabulary the model is trained on, read from the client's own
# realized sales: {'3EX', 'EX', 'FR', 'GD', 'VG', 'VG-GD'}.
#
# `VG-GD` is a REAL level with 405 stones (1.5% of training data), not a
# malformed value. The first version of this normaliser collapsed it to `VG`
# because it split on the separator and returned the cut grade — which meant a
# stone entering through the API priced 1.77 pts differently from the same stone
# entering through the Excel path. That is Trap 9 all over again, introduced by
# the very fix written to prevent it.
#
# So: combinations that the model has actually SEEN are preserved; combinations
# it has never seen still collapse to the cut grade, because an unseen category
# is what caused the original bug.
_CPS_TRAINED = {"3EX", "EX", "VG", "GD", "FR", "VG-GD"}


def normalize_cps(raw: str | None) -> str:
    """Canonicalise a cut/polish/symmetry code to the trained vocabulary.

    Accepts the separated forms a CRM is likely to send ("EX-EX-EX", "EX EX EX",
    "EX/EX/EX", "VG-EX-EX") as well as the already-canonical codes.

    Mirrors `_make_cps`: all three excellent -> "3EX", otherwise the CUT grade
    (the first component), because that is what the model was trained on.
    Unknown input returns "NA" — the honest "no cut information" value — rather
    than a guess, since guessing high is exactly how a VG stone gets priced
    like a 3EX.

    MUST BE IDEMPOTENT: every output ("3EX", "EX", "VG", "GD", "FR", "PR", "NA")
    is itself accepted and returned unchanged.
    """
    if raw is None:
        return "NA"
    s = str(raw).strip().upper()
    if not s or s in {"NAN", "NONE", "NA", "N/A", "-"}:
        return "NA"
    if s.replace(" ", "").replace("-", "").replace("/", "") in _TRIPLE_EX:
        return "3EX"
    if s in _CPS_GRADE:
        return _CPS_GRADE[s]
    # Separated forms: EX-EX-EX, EX EX EX, EX/EX/EX, VG-EX-EX, ...
    parts = [p for p in re.split(r"[-/,\s]+", s) if p]
    grades = [_CPS_GRADE.get(p) for p in parts]
    if grades and all(g is not None for g in grades):
        if len(grades) >= 3 and all(g == "EX" for g in grades[:3]):
            return "3EX"
        # Keep a two-grade combination if the model was actually trained on it
        # (e.g. VG-GD, 405 stones). Collapsing a level the model knows throws
        # away real signal and makes the API disagree with the Excel path.
        if len(grades) == 2:
            combined = f"{grades[0]}-{grades[1]}"
            if combined in _CPS_TRAINED:
                return combined
        return grades[0]        # the cut grade — what the model knows
    return "NA"


def normalize_shape(raw: str | None) -> str | None:
    """Canonicalise a shape code or spelling to the engine's `Shape_full`.

    ONLY known synonyms are converted. Anything else is returned EXACTLY as it
    arrived (stripped), never reshaped.

    It used to fall through to `s.title()`, which was wrong in the same way the
    CPS collapse was wrong: the client trades small-volume shapes whose trained
    names are not title-case — `S.MQ`, `DECA BRI`, `OLD MINE CL`,
    `FLAME STEP CUT`. Title-casing them produced `S.Mq`, `Deca Bri`, ... none of
    which match the training vocabulary, so every one of those stones scored
    zero history, was flagged `rare_shape` and dropped to the sparse fallback.

    The client's systems send the same strings the model trained on, so passing
    an unrecognised value through untouched is what keeps the two in agreement.
    A genuinely unknown shape still routes as rare, which is the honest answer
    for something we have no history on.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return _SHAPE_FULL.get(s.upper(), s)


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
