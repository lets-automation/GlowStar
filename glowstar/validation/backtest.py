"""Out-of-time backtest: baseline vs model, overall and per segment.

Train on sales strictly before the split date; test on sales on/after it
(brief Section 12). No forbidden features (enforced in the feature layer); the
target FDiscount is never an input. Reports discount-point and dollar errors,
+/-3 and +/-5 hit rates, interval calibration, and the baseline comparison so
improvement is provable, not asserted.

Run:  python -m glowstar.validation.backtest
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SETTINGS
from ..data.loaders import load_records, sold_stones
from ..features.build import build_features, get_target
from ..models.baseline import HierarchicalMedianModel
from ..models.gbm import QuantileGBM
from . import metrics as M


@dataclass
class SplitInfo:
    split_date: str
    n_train: int
    n_test: int


def time_split(sold: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    cut = pd.Timestamp(split_date)
    train = sold[sold["OrderDate_dt"] < cut].copy()
    test = sold[sold["OrderDate_dt"] >= cut].copy()
    return train, test, SplitInfo(split_date, len(train), len(test))


def recency_weights(train: pd.DataFrame, split_date: str, half_life_days: float = 60.0) -> np.ndarray:
    """Exponential recency weights: sales nearer the split count more.

    Measured to cut out-of-time bias on this data; the residual it cannot
    remove is what the market anchor (market layer) is for.
    """
    cut = pd.Timestamp(split_date)
    age = (cut - train["OrderDate_dt"]).dt.days.to_numpy().astype(float)
    return 0.5 ** (age / half_life_days)


def _segment_table(pred_model, pred_base, test, by: str, min_n: int = 25) -> pd.DataFrame:
    rows = []
    for seg, idx in test.groupby(by, observed=True).groups.items():
        sub = test.loc[idx]
        if len(sub) < min_n:
            continue
        mm = M.compute(pred_model[test.index.get_indexer(idx)], sub)
        bm = M.compute(pred_base[test.index.get_indexer(idx)], sub)
        rows.append({
            "segment": seg, "n": len(sub),
            "model_mae": round(mm.mae, 2), "base_mae": round(bm.mae, 2),
            "model_within5": round(mm.within5, 3), "base_within5": round(bm.within5, 3),
            "model_$medae": round(mm.dollar_medae, 0), "base_$medae": round(bm.dollar_medae, 0),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def run_backtest(split_date: str | None = None) -> dict:
    split_date = split_date or SETTINGS.backtest_split_date
    df, rep = load_records()
    sold = sold_stones(df, drop_outliers=True)
    train, test, info = time_split(sold, split_date)

    # --- Baseline (hierarchical median) ---
    base = HierarchicalMedianModel().fit(train)
    base_pred = base.predict(test)

    # --- Model (quantile GBM) ---
    # Freeze the market_month_index origin to the training epoch so train and
    # test share one time scale (no per-frame origin reset).
    month_base = train["MarketSheetDate_dt"].min()
    x_train, y_train = build_features(train, month_base), get_target(train)
    x_test = build_features(test, month_base)
    weights = recency_weights(train, split_date)
    gbm = QuantileGBM().fit(x_train, y_train, sample_weight=weights)
    lo, mid, hi = gbm.predict_interval(x_test)

    # --- Metrics ---
    m_model = M.compute(mid, test)
    m_base = M.compute(base_pred, test)
    calib = M.interval_calibration(lo, hi, test["FDiscount"].to_numpy())

    # Method split: how many test stones the baseline served via fallback.
    fb = base.predict_detailed(test)
    n_fallback = sum(p.is_fallback for p in fb)

    result = {
        "split": info.__dict__,
        "data_report": rep.summary(),
        "model": m_model.as_dict(),
        "baseline": m_base.as_dict(),
        "interval_target": gbm.hi_q - gbm.lo_q,
        "interval_empirical_coverage": round(calib, 3),
        "fallback_share_in_test": round(n_fallback / len(test), 3),
        "by_shape": _segment_table(mid, base_pred, test, "Shape_full"),
    }
    return result


def _fmt(title: str, m: dict) -> str:
    return (f"  {title:<10} n={m['n']:>5}  MAE={m['mae']:.2f}  MedAE={m['medae']:.2f}  "
            f"±3={m['within3']:.1%}  ±5={m['within5']:.1%}  "
            f"$MedAE={m['dollar_medae']:,.0f}  $MAE={m['dollar_mae']:,.0f}")


def main() -> None:
    r = run_backtest()
    print("=" * 78)
    print("GLOW STAR PRICING ENGINE — OUT-OF-TIME BACKTEST")
    print("=" * 78)
    print(r["data_report"])
    s = r["split"]
    print(f"\nSplit @ {s['split_date']}:  train={s['n_train']:,} sold  test={s['n_test']:,} sold")
    print(f"(target = FDiscount, discount off Rap; leakage-free features only)\n")
    print("OVERALL (discount points & USD per stone):")
    print(_fmt("MODEL", r["model"]))
    print(_fmt("BASELINE", r["baseline"]))
    lift = r["baseline"]["mae"] - r["model"]["mae"]
    print(f"\n  -> model beats baseline by {lift:.2f} MAE points "
          f"({lift / r['baseline']['mae']:.1%} lower error)")
    print(f"\nCONFIDENCE INTERVAL  target={r['interval_target']:.0%}  "
          f"empirical={r['interval_empirical_coverage']:.1%}  "
          f"(well-calibrated if empirical ~ target)")
    print(f"Fallback share in test: {r['fallback_share_in_test']:.1%}")
    print("\nPER-SHAPE (model vs baseline):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(r["by_shape"].to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
