"""Honest hyperparameter tuning on an INNER validation split (brief Section 12).

The trap with tuning is leaking the test window into model selection. We avoid
it with a strict nested split:

    inner_train  <  val_split        (fit candidates)
    inner_val    in [val_split, test_split)   (SELECT the config here)
    test         >= test_split        (report the winner ONCE, untouched by tuning)

Point accuracy is tuned with the fast median-quantile path (no conformal bands)
so a wide grid is cheap; the chosen config is then handed to the full engine for
the honest test-set number.

Run:  python -m glowstar.validation.tune
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import SETTINGS
from ..data.loaders import load_records, sold_stones
from ..features.build import build_features, get_target
from ..market.anchor import MarketTables, calibrate_offsets, anchor_predictions
from .backtest import time_split
from . import metrics as M

warnings.filterwarnings("ignore")

# Inner validation boundary (one month before the real test split by default).
VAL_SPLIT = "2026-04-01"


def _recency_w(tr: pd.DataFrame, half_life: float) -> np.ndarray:
    age = (tr["OrderDate_dt"].max() - tr["OrderDate_dt"]).dt.days.to_numpy().astype(float)
    return 0.5 ** (age / half_life)


def _fit_predict(tr: pd.DataFrame, ev: pd.DataFrame, tables: MarketTables, *,
                 lam: float, half_life: float, min_n: int, shrink_k: float,
                 gbm_kw: dict) -> np.ndarray:
    """Fast point path: median GBM (recency-weighted) + segment-anchored."""
    base = tr["MarketSheetDate_dt"].min()
    m = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.5, categorical_features="from_dtype",
        random_state=42, **gbm_kw)
    m.fit(build_features(tr, base), get_target(tr), sample_weight=_recency_w(tr, half_life))
    raw = m.predict(build_features(ev, base))
    cal = calibrate_offsets(tr, tables, min_n=min_n, shrink_k=shrink_k)
    return anchor_predictions(raw, ev, tables, cal, lam=lam)


# Search grid. Engine-level levers first; a couple of GBM knobs second.
GRID = {
    "lam": [0.20, 0.30, 0.35, 0.45],
    "half_life": [30.0, 45.0, 60.0],
    "min_n": [20, 25],
    "shrink_k": [40.0],
    "gbm": [
        {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31,
         "min_samples_leaf": 40, "l2_regularization": 1.0,
         "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 30},
        {"learning_rate": 0.05, "max_iter": 800, "max_leaf_nodes": 47,
         "min_samples_leaf": 60, "l2_regularization": 2.0,
         "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 30},
    ],
}


def run() -> dict:
    df, _ = load_records()
    sold = sold_stones(df, drop_outliers=True)
    tables = MarketTables.load()

    inner_train, rest, _ = time_split(sold, VAL_SPLIT)
    inner_val, _, _ = time_split(rest, SETTINGS.backtest_split_date)  # val = [VAL_SPLIT, test_split)
    val_actual = inner_val["FDiscount"].to_numpy()

    results = []
    for lam, hl, min_n, sk, gbm in itertools.product(
            GRID["lam"], GRID["half_life"], GRID["min_n"], GRID["shrink_k"], GRID["gbm"]):
        pred = _fit_predict(inner_train, inner_val, tables,
                            lam=lam, half_life=hl, min_n=min_n, shrink_k=sk, gbm_kw=gbm)
        mm = M.compute(pred, inner_val)
        results.append({
            "lam": lam, "half_life": hl, "min_n": min_n, "shrink_k": sk,
            "gbm": "lite" if gbm["max_iter"] == 600 else "deep",
            "val_mae": round(mm.mae, 3), "val_within5": round(mm.within5, 3),
            "val_bias": round(float(np.mean(pred - val_actual)), 3),
        })
    results.sort(key=lambda r: r["val_mae"])
    best = results[0]

    # Report the winner ONCE on the real out-of-time test set.
    train, test, info = time_split(sold, SETTINGS.backtest_split_date)
    gbm_kw = GRID["gbm"][0] if best["gbm"] == "lite" else GRID["gbm"][1]
    test_pred = _fit_predict(train, test, tables, lam=best["lam"],
                             half_life=best["half_life"], min_n=best["min_n"],
                             shrink_k=best["shrink_k"], gbm_kw=gbm_kw)
    tm = M.compute(test_pred, test)
    return {"results": results, "best": best, "split": info.__dict__,
            "test_mae": round(tm.mae, 3), "test_within5": round(tm.within5, 3),
            "test_bias": round(float(np.mean(test_pred - test["FDiscount"].to_numpy())), 3)}


def main() -> None:
    r = run()
    print("=" * 74)
    print("INNER-VALIDATION TUNING  (select on val, report on test once)")
    print("=" * 74)
    print(f"val window = [{VAL_SPLIT}, {SETTINGS.backtest_split_date})   "
          f"test = >= {SETTINGS.backtest_split_date}\n")
    print(f"{'lam':>5}{'hl':>5}{'min_n':>6}{'gbm':>6}{'val_MAE':>9}{'val_±5':>8}{'val_bias':>9}")
    for row in r["results"][:12]:
        print(f"{row['lam']:>5}{row['half_life']:>5.0f}{row['min_n']:>6}{row['gbm']:>6}"
              f"{row['val_mae']:>9}{row['val_within5']:>8.1%}{row['val_bias']:>9}")
    b = r["best"]
    print(f"\nBEST (by val MAE): lam={b['lam']} half_life={b['half_life']} "
          f"min_n={b['min_n']} shrink_k={b['shrink_k']} gbm={b['gbm']}")
    print(f"  -> HONEST TEST: MAE={r['test_mae']}  ±5={r['test_within5']:.1%}  "
          f"bias={r['test_bias']:+.2f}  (test n={r['split']['n_test']:,})")
    print("=" * 74)


if __name__ == "__main__":
    main()
