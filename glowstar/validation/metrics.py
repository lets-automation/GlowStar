"""Accuracy metrics in both discount-space and dollar-space (brief Section 12)."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    n: int
    mae: float            # mean abs error, discount points
    medae: float          # median abs error, discount points
    within3: float        # share within +/-3 discount points
    within5: float        # share within +/-5 discount points
    dollar_medae: float   # median abs error per stone, USD
    dollar_mae: float     # mean abs error per stone, USD

    def as_dict(self) -> dict:
        return asdict(self)


def discount_metrics(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float, float, float]:
    err = np.abs(pred - actual)
    return (
        float(np.mean(err)),
        float(np.median(err)),
        float(np.mean(err <= 3.0)),
        float(np.mean(err <= 5.0)),
    )


def dollar_errors(pred_disc: np.ndarray, df: pd.DataFrame) -> tuple[float, float]:
    """Translate a discount prediction to price and compare to FNetAmount.

    price = Rap * (1 + discount/100) * Weight  (the verified identity, Sec 4.1)
    """
    rap = df["Rap"].to_numpy()
    wt = df["Weight"].to_numpy()
    pred_price = rap * (1.0 + pred_disc / 100.0) * wt
    actual_price = df["FNetAmount"].to_numpy()
    err = np.abs(pred_price - actual_price)
    return float(np.median(err)), float(np.mean(err))


def compute(pred_disc: np.ndarray, df: pd.DataFrame, target: str = "FDiscount") -> Metrics:
    actual = df[target].to_numpy()
    mae, medae, w3, w5 = discount_metrics(pred_disc, actual)
    d_med, d_mean = dollar_errors(pred_disc, df)
    return Metrics(
        n=len(df), mae=mae, medae=medae, within3=w3, within5=w5,
        dollar_medae=d_med, dollar_mae=d_mean,
    )


def interval_calibration(lo: np.ndarray, hi: np.ndarray, actual: np.ndarray) -> float:
    """Empirical coverage: share of actuals inside [lo, hi]."""
    return float(np.mean((actual >= lo) & (actual <= hi)))
