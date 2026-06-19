"""Internal market-trend index (the directional, temporal market signal).

This is the "market research" correction the brief's north star depends on, and
it is *different* from the Uni cross-sectional anchor:

  * Uni anchor  -> WHERE the market is right now, by segment (a level).
  * This index  -> WHICH WAY and HOW FAST the market is moving over time (a
                   trend), so forward pricing doesn't lag a falling market.

It is a quality-ADJUSTED price index: we regress out the 4C/shape mix so the
index reflects genuine market movement, not a change in what happened to sell.
Built from the client's own realized sales (authentic, self-collected, no
external dependency). The macro RAPI/lab-grown context (context.py) is a
provenance-tagged external cross-check on top of this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from ..reference.normalize import CLARITY_ORDER

log = logging.getLogger(__name__)

_COLOR_ORD = {c: i for i, c in enumerate("DEFGHIJKLMN")}
_CLARITY_ORD = {c: i for i, c in enumerate(CLARITY_ORDER)}


def _quality_design(df: pd.DataFrame, encoder: OneHotEncoder, fit: bool):
    num = np.column_stack([
        np.log(df["Weight"].clip(lower=1e-3)),
        np.log(df["Rap"].clip(lower=1.0)),
        df["Color"].map(_COLOR_ORD).fillna(6).to_numpy(),
        df["Clarity"].map(_CLARITY_ORD).fillna(6).to_numpy(),
        (df["Shape_full"].str.lower() == "round").astype(float).to_numpy(),
    ])
    shapes = df["Shape_full"].astype("string").fillna("NA").to_numpy().reshape(-1, 1)
    oh = encoder.fit_transform(shapes) if fit else encoder.transform(shapes)
    return np.hstack([num, oh])


@dataclass
class MarketIndex:
    """Quality-adjusted monthly discount index + trend projection."""

    series: pd.Series = None          # index value (discount pts) by month period
    recent_slope: float = 0.0         # pts/month over the recent window
    last_period: pd.Period = None

    _ridge: Ridge = None
    _encoder: OneHotEncoder = None

    def fit(self, sold: pd.DataFrame, recent_months: int = 3) -> "MarketIndex":
        self._encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        x = _quality_design(sold, self._encoder, fit=True)
        y = sold["FDiscount"].to_numpy()
        self._ridge = Ridge(alpha=10.0).fit(x, y)

        resid = y - self._ridge.predict(x)
        month = sold["OrderDate_dt"].dt.to_period("M")
        self.series = pd.Series(resid, index=month).groupby(level=0).median().sort_index()
        self.last_period = self.series.index.max()

        # Robust recent slope (pts/month) via least squares over the recent window.
        tail = self.series.iloc[-recent_months:] if len(self.series) >= 2 else self.series
        if len(tail) >= 2:
            t = np.arange(len(tail))
            self.recent_slope = float(np.polyfit(t, tail.to_numpy(), 1)[0])
        return self

    def level(self, period: pd.Period | str) -> float:
        """Index level for a month (projected if beyond the observed series)."""
        p = pd.Period(period, freq="M") if not isinstance(period, pd.Period) else period
        if self.series is not None and p in self.series.index:
            return float(self.series.loc[p])
        return self.project(p)

    def project(self, period: pd.Period | str) -> float:
        """Extrapolate beyond the last observed month using the recent slope.

        Labelled as an estimate by callers; not presented as observed data.
        """
        p = pd.Period(period, freq="M") if not isinstance(period, pd.Period) else period
        if self.series is None or self.last_period is None:
            return 0.0
        steps = (p - self.last_period).n
        if steps <= 0:
            return float(self.series.iloc[-1])
        return float(self.series.iloc[-1] + self.recent_slope * steps)

    def drift(self, from_period, to_period) -> float:
        """Change in market level from one month to another (to - from)."""
        return self.level(to_period) - self.level(from_period)

    def as_dict(self) -> dict:
        return {
            "monthly_index": {str(k): round(v, 2) for k, v in self.series.items()},
            "recent_slope_pts_per_month": round(self.recent_slope, 3),
            "last_observed_month": str(self.last_period),
            "direction": "softening" if self.recent_slope < -0.3
                         else "firming" if self.recent_slope > 0.3 else "flat",
        }
