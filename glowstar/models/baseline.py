"""Hierarchical-median baseline + rare-shape fallback (brief Sections 7.6, 12).

This serves two roles:
  1. The transparent BASELINE the model must beat (segment-median discount).
  2. The rare-shape / sparse-segment FALLBACK: when a stone's segment has too
     few training examples, prediction backs off to the coarsest segment with
     enough support, the interval widens, and method is marked `fallback`.

It predicts in discount-space (FDiscount), reports the segment level used and
the supporting count, so every number has a basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SETTINGS

# Most specific -> least specific. `bracket_index` is added by the caller's
# feature step; here we recompute simple keys from raw columns for transparency.
_LEVELS: tuple[tuple[str, ...], ...] = (
    ("Shape_full", "size_band", "Color", "Clarity"),
    ("Shape_full", "size_band", "Color"),
    ("Shape_full", "size_band"),
    ("Shape_full",),
    (),  # global
)

# Coarse size bands for the backoff hierarchy (kept coarse on purpose so the
# fallback still has support for rare shapes).
_SIZE_EDGES = np.array([0.0, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00, 3.00, 5.00, np.inf])


def _size_band(weight: pd.Series) -> pd.Series:
    idx = np.digitize(weight.to_numpy(), _SIZE_EDGES[1:-1])
    return pd.Series(idx, index=weight.index).astype("int")


@dataclass
class Prediction:
    """A single discount prediction with its basis."""

    discount: float
    level: int            # index into _LEVELS used (0 = most specific)
    count: int            # supporting training stones
    spread: float         # IQR-based spread at that level (for interval width)
    is_fallback: bool


class HierarchicalMedianModel:
    """Segment-median discount with hierarchical backoff."""

    def __init__(self, min_samples: int | None = None):
        self.min_samples = min_samples or SETTINGS.min_segment_samples
        self._tables: list[dict] = []
        self._global_median: float = 0.0
        self._global_spread: float = 0.0

    def fit(self, df: pd.DataFrame, target: str = "FDiscount") -> "HierarchicalMedianModel":
        work = df.copy()
        work["size_band"] = _size_band(work["Weight"])
        self._tables = []
        for keys in _LEVELS:
            if not keys:
                self._tables.append({})
                continue
            grp = work.groupby(list(keys), observed=True)[target]
            stats = grp.agg(["median", "count",
                             lambda s: s.quantile(0.75) - s.quantile(0.25)])
            stats.columns = ["median", "count", "iqr"]
            self._tables.append(stats.to_dict("index"))
        self._global_median = float(work[target].median())
        self._global_spread = float(work[target].quantile(0.75) - work[target].quantile(0.25))
        return self

    def _predict_row(self, row: pd.Series) -> Prediction:
        for level, keys in enumerate(_LEVELS):
            if not keys:
                break
            key = tuple(row[k] for k in keys)
            key = key[0] if len(key) == 1 else key
            rec = self._tables[level].get(key)
            if rec and rec["count"] >= self.min_samples:
                iqr = rec["iqr"] if not np.isnan(rec["iqr"]) else self._global_spread
                return Prediction(
                    discount=float(rec["median"]), level=level,
                    count=int(rec["count"]), spread=float(iqr),
                    is_fallback=level >= 2,   # below color granularity = fallback regime
                )
        return Prediction(
            discount=self._global_median, level=len(_LEVELS) - 1,
            count=0, spread=self._global_spread, is_fallback=True,
        )

    def predict_detailed(self, df: pd.DataFrame) -> list[Prediction]:
        work = df.copy()
        work["size_band"] = _size_band(work["Weight"])
        return [self._predict_row(r) for _, r in work.iterrows()]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([p.discount for p in self.predict_detailed(df)])
