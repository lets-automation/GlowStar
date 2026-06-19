"""Market anchor: re-center pricing toward the CURRENT market level and apply
soft-attribute (BGM/milky/shade) discounts learned from the Uni feed.

Why this exists (proven empirically): a model trained on the client's past
sales carries the *past* discount level. In a moving market the realized level
drifts, so the model is biased on recent stones — and the bias concentrates in
liquid commercial goods (rounds). The Uni market snapshot is ~contemporaneous
with the present, so its per-segment median discount carries the *current*
level. Blending the model with a calibrated market anchor pulls the level back
to now; the soft-attribute deltas correct for quality signals the client's own
data doesn't record yet.

Reference mismatch handled honestly: Uni discounts are *asking* prices from a
different dealer; the client's FDiscount is *realized*. We learn the
asking->realized offset per the most recent training window and apply it, rather
than assuming the two are on the same scale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR
from .aggregate_bulk import _milky_severity, _shade_class
from .segments import segment_keys

log = logging.getLogger(__name__)


@dataclass
class MarketTables:
    """Loaded market artifacts: segment medians + soft-attribute deltas."""

    segments: dict[str, dict]
    bgm: dict

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "MarketTables":
        d = artifacts_dir or ARTIFACTS_DIR
        segments = json.loads((d / "market_segments.json").read_text(encoding="utf-8"))
        bgm = json.loads((d / "bgm_discounts.json").read_text(encoding="utf-8"))
        return cls(segments=segments, bgm=bgm)

    # --- market level (with hierarchical backoff) ---
    def market_median(self, shape: str, weight: float, color: str, clarity: str,
                      min_n: int = 20) -> float | None:
        for key in segment_keys(shape, weight, color, clarity):
            name = "|".join(map(str, key)) if key else "__global__"
            rec = self.segments.get(name)
            if rec and rec["n"] >= min_n and not np.isnan(rec["median_discount"]):
                return float(rec["median_discount"])
        return None

    # --- soft-attribute delta (extra discount vs clean) ---
    def soft_delta(self, milky_raw: str | None, shade_raw: str | None) -> float:
        delta = 0.0
        m = _milky_severity(milky_raw)
        s = _shade_class(shade_raw)
        if m != "none":
            delta += self.bgm["by_milky"].get(m, {}).get("delta_vs_clean", 0.0)
        if s == "negative":
            delta += self.bgm["by_shade"].get("negative", {}).get("delta_vs_clean", 0.0)
        return float(delta)


def market_series(df: pd.DataFrame, tables: MarketTables) -> pd.Series:
    """Per-stone market median discount (NaN where no segment has support)."""
    vals = [
        tables.market_median(r.Shape_full, r.Weight, r.Color, r.Clarity)
        for r in df.itertuples()
    ]
    return pd.Series([np.nan if v is None else v for v in vals], index=df.index)


def calibrate_offset(train: pd.DataFrame, tables: MarketTables,
                     recent_days: int = 45) -> float:
    """Median (FDiscount - market_median) over the most recent training window.

    This is the asking->realized (and residual time) offset. Using only the
    recent window keeps the calibration close to the snapshot's epoch.
    """
    cut = train["OrderDate_dt"].max() - pd.Timedelta(days=recent_days)
    recent = train[train["OrderDate_dt"] >= cut]
    mkt = market_series(recent, tables)
    diff = (recent["FDiscount"] - mkt).dropna()
    return float(diff.median()) if len(diff) else 0.0


def anchor_predictions(model_pred: np.ndarray, df: pd.DataFrame, tables: MarketTables,
                       offset: float, lam: float = 0.35) -> np.ndarray:
    """Blend model prediction with the calibrated market anchor.

    final = (1-lam)*model + lam*(market_median + offset), per stone, only where
    a market segment has support; otherwise the model prediction is unchanged.
    """
    mkt = market_series(df, tables).to_numpy()
    anchor = mkt + offset
    out = model_pred.copy().astype(float)
    has = ~np.isnan(anchor)
    out[has] = (1.0 - lam) * model_pred[has] + lam * anchor[has]
    return out
