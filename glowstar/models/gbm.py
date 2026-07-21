"""Quantile gradient-boosting pricing model (brief Sections 7.1, 7.8).

Predicts FDiscount (discount off Rap) and a calibrated confidence interval via
three HistGradientBoostingRegressors trained on the lower / median / upper
quantiles. Native categorical handling (pandas 'category' dtype) means rare
levels are tolerated without manual encoding.

The interval is genuine quantile regression; the backtest harness measures its
empirical coverage so we can report calibration honestly rather than assert it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import SETTINGS


class QuantileGBM:
    """Lower/median/upper quantile GBMs for discount with intervals."""

    # DELIBERATELY EMPTY — do not add color/clarity here. It looks like an obvious
    # win and it is wrong.
    #
    # The reasoning that fails: "a worse colour must carry a deeper discount, so
    # constrain discount to decrease in color_ordinal." A sweep shows ~40% of
    # adjacent colour pairs 'inverted' and the constraint drives that to ~1% for
    # ~0.02 MAE — it looks free and virtuous.
    #
    # But the target is a DISCOUNT OFF RAP, and Rap ALREADY prices colour and
    # clarity. The discount is the DEVIATION from Rap, and it is genuinely NOT
    # monotone: measured on the client's own realized sales, 47.7% of well-supported
    # adjacent colour pairs have the worse colour at a SHALLOWER discount. Their real
    # book, Round/3EX/VS1/0.30-0.60ct:
    #     D -46.2   E -46.9   F -44.0   G -44.1   H -45.8   I -42.1   J -44.5
    # Commercial goods (F-H) trade shallower off Rap than hard-to-move D-E. The
    # 'inversions' are the market, not a bug — the model was learning them correctly.
    # By contrast PRICE ($/ct) is far more monotone (17.9% reversals), which is the
    # invariant the desk actually judges.
    #
    # Constraining the discount imposes a rule the client's own market breaks half
    # the time. If you ever want this, constrain PRICE, and prove it on their sales
    # first. (Same reason Color/Clarity stay in CATEGORICAL_FEATURES.)
    MONOTONIC: dict[str, int] = {}

    def __init__(self, coverage: float | None = None, random_state: int = 42):
        cov = coverage if coverage is not None else SETTINGS.interval_coverage
        self.lo_q = (1.0 - cov) / 2.0
        self.hi_q = 1.0 - self.lo_q
        self._rs = random_state
        self.models: dict[str, HistGradientBoostingRegressor] = {}

    def _monotonic_cst(self, x: pd.DataFrame) -> list[int]:
        """Per-column constraint vector, positional to `x.columns`."""
        return [self.MONOTONIC.get(c, 0) for c in x.columns]

    def _make(self, quantile: float, monotonic_cst: list[int] | None = None
              ) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            categorical_features="from_dtype",
            monotonic_cst=monotonic_cst,
            random_state=self._rs,
        )

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "QuantileGBM":
        """Fit the three quantile models.

        `sample_weight` lets the caller down-weight older sales (recency
        weighting) so the model tracks the moving market level — measured to
        reduce out-of-time bias on this data.
        """
        cst = self._monotonic_cst(x)
        for name, q in (("lo", self.lo_q), ("mid", 0.5), ("hi", self.hi_q)):
            self.models[name] = self._make(q, cst).fit(x, y, sample_weight=sample_weight)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Point (median-quantile) discount prediction."""
        return self.models["mid"].predict(x)

    def predict_interval(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, median, upper) discount predictions.

        Quantile crossing (lo > hi) is repaired by sorting the three estimates
        per row so the interval is always well-ordered.
        """
        lo = self.models["lo"].predict(x)
        mid = self.models["mid"].predict(x)
        hi = self.models["hi"].predict(x)
        stacked = np.sort(np.vstack([lo, mid, hi]), axis=0)
        return stacked[0], stacked[1], stacked[2]
