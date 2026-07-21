"""Learn the client's list-pricing STRUCTURE from their Master grid.

The grid covers standard cells exactly, but not every stone (odd sizes, rarer
combos, shapes the grid omits). This model is trained on the grid cells
(attributes -> the client's list discount) so it can predict a GRID-CONSISTENT
list price for a stone the grid doesn't explicitly contain — i.e. price it the
way the client's own sheet would, without copying any single cell.

This is deliberately SEPARATE from the realized-sale model (models/engine.py):
grid values are LIST prices; mixing them into the realized model biases it (the
promotion gate proved this). Here they are exactly the right labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..reference.normalize import CLARITY_ORDER

log = logging.getLogger(__name__)

_COLOR_ORDER = ("D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N")
_COLOR_ORD = {c: i for i, c in enumerate(_COLOR_ORDER)}
_CLARITY_ORD = {c: i for i, c in enumerate(CLARITY_ORDER)}

_CAT = ("shape", "cut", "fluor")
_NUM = ("color_ordinal", "clarity_ordinal", "size", "log_size")


def _grid_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["shape"] = df["shape"].astype("string").str.upper().fillna("NA").astype("category")
    x["cut"] = df["cut"].astype("string").str.upper().fillna("NA").astype("category")
    x["fluor"] = df["fluor"].astype("string").str.upper().fillna("NON").astype("category")
    x["color_ordinal"] = df["color"].astype("string").str.upper().map(_COLOR_ORD).astype("float64")
    x["clarity_ordinal"] = df["clarity"].astype("string").str.upper().map(_CLARITY_ORD).astype("float64")
    x["size"] = df["size"].astype("float64")
    x["log_size"] = np.log(df["size"].clip(lower=1e-3))
    return x


def cells_to_frame(cells: list[dict]) -> pd.DataFrame:
    """Grid cells -> a training frame (one row per (cell, shape))."""
    rows = []
    for c in cells:
        disc = c.get("discount")
        if disc is None:
            continue
        try:
            size = (float(c["minWeight"]) + float(c["maxWeight"])) / 2.0
        except (TypeError, ValueError, KeyError):
            continue
        for sh in (c.get("shape") or []):
            rows.append({"shape": sh, "color": c.get("color"), "clarity": c.get("clarity"),
                         "cut": c.get("cut"), "fluor": c.get("fluorescence"),
                         "size": size, "discount": float(disc)})
    return pd.DataFrame(rows)


@dataclass
class GridModel:
    model: HistGradientBoostingRegressor

    @classmethod
    def fit(cls, cells: list[dict]) -> "GridModel":
        df = cells_to_frame(cells)
        if len(df) < 200:
            raise ValueError(f"Too few grid cells to fit a grid model ({len(df)}).")
        m = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.08, max_iter=400, max_leaf_nodes=63,
            min_samples_leaf=20, l2_regularization=1.0,
            categorical_features="from_dtype", random_state=42)
        m.fit(_grid_features(df), df["discount"].to_numpy())
        log.info("Fitted grid model on %d cell-rows.", len(df))
        return cls(model=m)

    def predict(self, shape, weight, color, clarity, cps, fluorescence) -> float:
        df = pd.DataFrame([{"shape": shape, "color": color, "clarity": clarity,
                            "cut": cps, "fluor": fluorescence, "size": float(weight)}])
        return float(self.model.predict(_grid_features(df))[0])
