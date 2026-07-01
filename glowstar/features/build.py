"""Build the model feature matrix from records (brief Sections 7.3, 7.4).

Hard rules enforced here:
  * Only attributes knowable at pricing time are used.
  * Transaction-derived columns are NEVER used (leakage guard raises if any
    appear in the matrix). This protects the honest accuracy claim.
  * AvailableDays / Ageing are excluded — they are only known after a stone has
    sat/sold, so they would leak into a price-at-listing prediction. They belong
    to the inventory engine, not the pricing model.
  * Bracket membership AND exact weight are both encoded so the model can learn
    Rapaport's size "cliffs" without naive interpolation across them.

Soft attributes (BGM/milky/shade/eye-clean) are not in records.json yet; the
matrix leaves clean slots (`SOFT_FEATURES`) for them to be joined in later
(brief Section 5) without reshaping the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.loaders import FORBIDDEN_FEATURES
from ..reference.normalize import CLARITY_ORDER

TARGET = "FDiscount"

# Categorical features (native categorical handling in HistGradientBoosting).
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Shape_full", "Color", "Clarity", "CPS", "Fluorescence", "Lab", "Location",
)

# Numeric features.
NUMERIC_FEATURES: tuple[str, ...] = (
    "Weight", "log_weight", "Rap", "log_rap",
    "color_ordinal", "clarity_ordinal", "bracket_index", "is_round",
    "market_month_index",
)

# Soft-attribute slots — empty until the CRM/market join supplies them.
SOFT_FEATURES: tuple[str, ...] = (
    "milky_severity", "shade_class", "eye_clean", "is_bgm",
)

# CPS (Cut/Polish/Symmetry) training vocabulary — the ONLY values the model has
# ever seen in records.json. A categorical level outside this set is unknown to
# the GBM, which then SILENTLY DROPS the cut signal and prices the stone at the
# base rate (e.g. a 'VG-EX' stone gets priced like a 3EX — verified ~7 pts too
# shallow). So every CPS is clamped to this vocabulary at the feature boundary,
# regardless of how the stone entered (GIA export, CRM/service, combined codes).
_CPS_VOCAB: frozenset[str] = frozenset({"3EX", "EX", "VG", "GD", "VG-GD", "FR"})
_CUT_CANON: dict[str, str] = {
    "3EX": "3EX", "EX": "EX", "EXCELLENT": "EX", "ID": "EX", "IDEAL": "EX",
    "VG": "VG", "VERY GOOD": "VG", "GD": "GD", "GOOD": "GD", "G": "GD",
    "FR": "FR", "FAIR": "FR", "PR": "FR", "POOR": "FR",
}


def normalize_cps(raw) -> str:
    """Clamp any Cut/Polish/Symmetry representation to the training vocabulary.

    Known values pass through unchanged (incl. the combined 'VG-GD' the model was
    trained on). For an unknown combined code (e.g. 'VG-EX', 'EX-VG', 'VG-VG-X')
    the CUT grade leads (GIA orders Cut-Polish-Symmetry), so the first token maps
    to the cut tier the market actually prices on. Unparseable -> 'NA' (missing),
    never a silent unseen level."""
    s = str(raw if raw is not None else "").upper().strip()
    if not s or s in ("NAN", "NONE", "NA"):
        return "NA"
    if s in _CPS_VOCAB:
        return s
    first = s.split("-")[0].strip()
    return _CUT_CANON.get(first, _CUT_CANON.get(first.lstrip("3").strip(), "NA"))


# Color ordinal: D(best)=0 .. N=10. Fancy/cape colors -> NaN (treated as missing
# by the GBM, and such stones route to the fallback anyway).
_COLOR_ORDER = ("D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N")
_COLOR_ORD = {c: i for i, c in enumerate(_COLOR_ORDER)}
_CLARITY_ORD = {c: i for i, c in enumerate(CLARITY_ORDER)}

# Lower edges of the Rapaport size brackets (incl. the 6-9.99 gap as its own
# bin and 10.00+ as the top bin). Used to encode bracket membership.
_BRACKET_EDGES = np.array(
    [0.01, 0.04, 0.08, 0.15, 0.18, 0.23, 0.30, 0.40, 0.50, 0.70, 0.90,
     1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 6.00, 10.00, 11.00]
)


@dataclass(frozen=True)
class FeatureSpec:
    """The columns the model consumes, split by kind."""

    categorical: tuple[str, ...] = CATEGORICAL_FEATURES
    numeric: tuple[str, ...] = NUMERIC_FEATURES

    @property
    def all(self) -> list[str]:
        return [*self.numeric, *self.categorical]


FEATURE_SPEC = FeatureSpec()


def _market_month_index(df: pd.DataFrame, base: pd.Timestamp | None = None) -> pd.Series:
    """Months since a FIXED base MarketSheetDate — the pricing-time clock.

    Uses the listing date (knowable at pricing time), NOT OrderDate (the sale
    date, which is the label's timestamp and would leak in an out-of-time split).

    `base` MUST be frozen at training time and reused for every later call
    (test, conformal, serving). If it were recomputed per call as `df.min()`,
    the train and test matrices would sit on different origins, and a single-row
    serving frame would always score `month_index = 0` (treated as the earliest
    period) — a silent train/serve skew. The engine passes its training base.
    """
    d = df["MarketSheetDate_dt"]
    if base is None:
        base = d.min()
    return ((d - base).dt.days / 30.44).astype("float64")


def build_features(df: pd.DataFrame, month_base: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return the model feature matrix X (no target, no forbidden columns).

    `month_base` freezes the `market_month_index` origin to the training epoch so
    train/test/serve all share one time scale (see `_market_month_index`).
    """
    x = pd.DataFrame(index=df.index)

    # Numeric.
    x["Weight"] = df["Weight"].astype("float64")
    x["log_weight"] = np.log(df["Weight"].clip(lower=1e-3))
    x["Rap"] = df["Rap"].astype("float64")
    x["log_rap"] = np.log(df["Rap"].clip(lower=1.0))
    x["color_ordinal"] = df["Color"].map(_COLOR_ORD).astype("float64")
    x["clarity_ordinal"] = df["Clarity"].map(_CLARITY_ORD).astype("float64")
    x["bracket_index"] = np.digitize(df["Weight"].to_numpy(), _BRACKET_EDGES).astype("float64")
    x["is_round"] = (df["Shape_full"].str.strip().str.lower() == "round").astype("float64")
    x["market_month_index"] = _market_month_index(df, month_base)

    # Categorical (pandas 'category' dtype -> native GBM handling, robust to
    # unseen/rare levels). CPS is first clamped to the training vocabulary so an
    # unseen cut code can never silently drop the cut signal (see normalize_cps).
    for col in CATEGORICAL_FEATURES:
        s = df[col].map(normalize_cps) if col == "CPS" else df[col]
        x[col] = s.astype("string").fillna("NA").astype("category")

    _assert_no_leakage(x)
    return x


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return the modeling target (FDiscount), as float."""
    return df[TARGET].astype("float64")


# The complete set of columns the feature matrix is ever allowed to contain.
# Enforced as a WHITELIST (not just a forbidden blacklist) so any unexpected
# column — a transaction leak, a typo, or a future careless edit — trips the
# guard, instead of relying on the matrix never being built from raw df columns.
_ALLOWED_FEATURES: frozenset[str] = frozenset(
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + SOFT_FEATURES
)


def _assert_no_leakage(x: pd.DataFrame) -> None:
    """Raise if any transaction-derived column leaked in, or any column is not on
    the explicit feature whitelist (a stricter, load-bearing guard)."""
    leaked = FORBIDDEN_FEATURES.intersection(x.columns)
    if leaked:
        raise AssertionError(
            f"Leakage guard tripped — forbidden features in matrix: {sorted(leaked)}"
        )
    unexpected = set(x.columns) - _ALLOWED_FEATURES
    if unexpected:
        raise AssertionError(
            f"Feature whitelist tripped — columns not allowed in matrix: {sorted(unexpected)}"
        )
