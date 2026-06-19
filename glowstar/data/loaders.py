"""Load and validate the inventory/sales records (brief Sections 4, 6).

The loader is strict and honest: it verifies the schema, checks the exact
price identity that the brief guarantees, isolates outliers (logged, not
silently dropped), and returns a structured report alongside the data so any
data-quality issue is visible rather than buried.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PATHS, SETTINGS

log = logging.getLogger(__name__)

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
