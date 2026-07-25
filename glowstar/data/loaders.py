"""Load and validate the inventory/sales records (brief Sections 4, 6).

The loader is strict and honest: it verifies the schema, checks the exact
price identity that the brief guarantees, isolates outliers (logged, not
silently dropped), and returns a structured report alongside the data so any
data-quality issue is visible rather than buried.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS, SETTINGS

log = logging.getLogger(__name__)

# Severity ordinals for the client's live `BgmComments` field (e.g.
# "No BROWN LIGHT MILKY"). Ordinal (not one-hot) so the tree learns the monotone
# price impact from very few Medium/Heavy examples. Unknown -> NaN (HGB-native).
_BGM_LEVEL = {"NO": 0.0, "NONE": 0.0, "FAINT": 1.0, "SLIGHT": 1.0, "LIGHT": 1.0,
              "MEDIUM": 2.0, "HEAVY": 3.0}


def parse_bgm_comments(text) -> tuple[float, float]:
    """(milky_ord, brown_ord) from a `BgmComments` string; (nan, nan) if absent.

    milky:  No=0 / Light=1 / Medium=2 / Heavy=3   (verified monotone in the
            client's own realized discounts: 0 / -5 / -16 / -20 pts).
    brown:  No=0 / Faint,Light=1 / Medium=2 / Heavy=3.
    A physical inspection attribute known at listing time — a legitimate feature,
    NOT a transaction outcome (so not leakage)."""
    s = str(text or "").upper().strip()
    if not s:
        return float("nan"), float("nan")
    bm = re.search(r"(NO|NONE|FAINT|SLIGHT|LIGHT|MEDIUM|HEAVY)\s+BROWN", s)
    mm = re.search(r"(NO|NONE|FAINT|SLIGHT|LIGHT|MEDIUM|HEAVY)\s+MILKY", s)
    brown = _BGM_LEVEL.get(bm.group(1), float("nan")) if bm else float("nan")
    milky = _BGM_LEVEL.get(mm.group(1), float("nan")) if mm else float("nan")
    return milky, brown


# The client's STRUCTURED tinge fields (added to the inventory API 2026-07).
# Codes are <severity><attribute>: NO / F=Faint / L=Light / M=Medium / H=Heavy,
# e.g. LBR = Light Brown, MML = Medium Milky, HMT = Heavy tint (Shade).
# These supersede `BgmComments`, which only ever carried BROWN and MILKY — SHADE
# (tint) and GREEN were invisible to us before. Measured on held-out sales, adding
# shade+green cut error on TINGED stones from 3.93 -> 3.41 MAE (-13%) while leaving
# clean stones unchanged, which is exactly the expected shape of the effect.
# Verified against the client's own realized sales, controlled for 4C cell:
#   brown  LBR -5.0 / MBR -8.6      milky LML -5.4 / MML -6.0
#   shade  LMT -4.0 / MMT -9.0      green LGR -4.5 (only 8 stones — rarely material)
_TINGE_LEVEL = {"NO": 0.0, "NONE": 0.0, "F": 1.0, "L": 1.0, "M": 2.0, "H": 3.0}
_TINGE_FIELDS: dict[str, str] = {
    "Brown": "brown_ord", "Milky": "milky_ord", "Shade": "shade_ord", "Green": "green_ord",
}


def parse_tinge(raw, suffix: str) -> float:
    """Severity ordinal from a structured tinge code (e.g. 'LBR' + 'BR' -> 1.0).

    `suffix` is the attribute code (BR/ML/MT/GR). Returns NaN when absent or
    unrecognised — never a silent 0.0, which would assert "clean" on unknown data
    and is the exact assumption that over-prices a tinged stone.
    """
    s = str(raw or "").upper().strip()
    if not s or s in ("NAN", "NONE"):
        return float("nan")
    if s in ("NO", "NONE"):
        return 0.0
    if not s.endswith(suffix):
        return float("nan")
    return _TINGE_LEVEL.get(s[: -len(suffix)], float("nan"))

# Columns guaranteed by the verified schema (brief Section 4).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "StoneId", "Shape", "Shape_full", "Weight", "Color", "Clarity",
    "Fluorescence", "CPS", "Lab", "Status", "Rap", "Discount", "FDiscount",
    "NetAmount", "FNetAmount", "OrderDate", "MarketSheetDate", "CreatedDate",
    "AvailableDays", "Ageing",
)

VALID_STATUS = {"Sold", "Stock", "Transit"}

# Transaction-derived columns that must never become model features
# (brief Section 7.4). Centralized here so the feature layer can assert on them.
FORBIDDEN_FEATURES: frozenset[str] = frozenset({
    "BasePriceDiscount", "Discount", "NetAmount", "PerCarat",
    "FAmount", "FPerCarat", "FNetAmount", "FDiscount",
})

_DATE_COLUMNS = ("CreatedDate", "MarketSheetDate", "OrderDate")


@dataclass
class RecordsReport:
    """Validation summary for a records load. Surfaced, never hidden."""

    n_total: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    missing_columns: list[str] = field(default_factory=list)
    identity_checked: int = 0
    identity_mismatches: int = 0
    identity_max_abs_err: float = 0.0
    fdiscount_outliers: int = 0
    unknown_status_rows: int = 0

    def summary(self) -> str:
        return (
            f"records: {self.n_total} rows {self.status_counts} | "
            f"identity {self.identity_checked - self.identity_mismatches}/"
            f"{self.identity_checked} exact (max abs err ${self.identity_max_abs_err:.4f}) | "
            f"FDiscount outliers flagged: {self.fdiscount_outliers}"
        )


def load_records(path: Path | None = None) -> tuple[pd.DataFrame, RecordsReport]:
    """Load records.json into a validated DataFrame plus a report.

    Adds derived columns:
      - parsed datetime columns (`*_dt`)
      - `is_outlier` (FDiscount out of [fdiscount_min, fdiscount_max] for sold)
      - `identity_abs_err` (|FNetAmount - Rap*(1+FDiscount/100)*Weight|) for sold
    """
    path = Path(path) if path else PATHS.records
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    df = pd.DataFrame(raw)
    report = RecordsReport(n_total=len(df))

    report.missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if report.missing_columns:
        raise ValueError(f"records.json missing required columns: {report.missing_columns}")

    # Status hygiene.
    report.status_counts = df["Status"].value_counts().to_dict()
    report.unknown_status_rows = int((~df["Status"].isin(VALID_STATUS)).sum())

    # Parse dates (tz-naive UTC; the source stamps are all '...Z').
    for col in _DATE_COLUMNS:
        df[f"{col}_dt"] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)

    # Tinge (Brown / Milky / Shade / Green) severity ordinals.
    #
    # PREFER the client's STRUCTURED fields (added to the API 2026-07, ~99%
    # populated). Fall back to parsing the legacy free-text `BgmComments` only for
    # older snapshots that predate them. This matters: BgmComments carries ONLY
    # brown+milky, so shade (tint) and green were previously invisible — and both
    # are genuinely priced (shade MMT -9.0 pts on the client's own sales).
    _SUFFIX = {"Brown": "BR", "Milky": "ML", "Shade": "MT", "Green": "GR"}
    for src, col in _TINGE_FIELDS.items():
        if src in df.columns:
            df[col] = df[src].map(lambda v, s=_SUFFIX[src]: parse_tinge(v, s))
        else:
            df[col] = np.nan
    # Legacy backfill: only where the structured field was absent/unparseable.
    if "BgmComments" in df.columns and (df["milky_ord"].isna().any() or df["brown_ord"].isna().any()):
        parsed = df["BgmComments"].map(parse_bgm_comments)
        legacy_milky = pd.Series([p[0] for p in parsed], index=df.index)
        legacy_brown = pd.Series([p[1] for p in parsed], index=df.index)
        df["milky_ord"] = df["milky_ord"].fillna(legacy_milky)
        df["brown_ord"] = df["brown_ord"].fillna(legacy_brown)

    sold = df["Status"] == "Sold"

    # Verify the exact price identity for sold stones (brief Section 4.1):
    #   FNetAmount == Rap * (1 + FDiscount/100) * Weight
    implied = df["Rap"] * (1 + df["FDiscount"] / 100.0) * df["Weight"]
    df["identity_abs_err"] = np.where(sold, (df["FNetAmount"] - implied).abs(), np.nan)
    sold_err = df.loc[sold, "identity_abs_err"]
    report.identity_checked = int(sold.sum())
    # Allow a cent of float rounding; anything larger is a real mismatch.
    report.identity_mismatches = int((sold_err > 0.01).sum())
    report.identity_max_abs_err = float(sold_err.max()) if len(sold_err) else 0.0

    # Outlier guardrails on the modeling target (brief Section 4.3).
    df["is_outlier"] = False
    df.loc[sold, "is_outlier"] = (
        (df.loc[sold, "FDiscount"] > SETTINGS.fdiscount_max)
        | (df.loc[sold, "FDiscount"] < SETTINGS.fdiscount_min)
    )
    report.fdiscount_outliers = int(df["is_outlier"].sum())

    log.info(report.summary())
    return df, report


def sold_stones(df: pd.DataFrame, *, drop_outliers: bool = True) -> pd.DataFrame:
    """Return the sold subset used for training/backtest.

    Outliers are excluded from modeling by default but remain in the full frame
    (logged via the report) — never silently dropped from the source data.
    """
    out = df[df["Status"] == "Sold"].copy()
    if drop_outliers:
        out = out[~out["is_outlier"]]
    return out


def stock_stones(df: pd.DataFrame) -> pd.DataFrame:
    """Return the live inventory subset (Status == 'Stock')."""
    return df[df["Status"] == "Stock"].copy()
