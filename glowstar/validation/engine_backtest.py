"""Final out-of-time backtest of the full PricingEngine (brief Section 12).

Reports the production engine's accuracy: leakage-free, recency-weighted,
market-anchored, conformally calibrated, with fallback routing. Compares to the
transparent baseline and breaks results out per shape.

Run:  python -m glowstar.validation.engine_backtest
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from ..config import SETTINGS
from ..data.loaders import load_records, sold_stones
from ..models.baseline import HierarchicalMedianModel
from ..models.engine import PricingEngine, EngineConfig
from . import metrics as M
from .backtest import time_split


def run(split_date: str | None = None) -> dict:
    split_date = split_date or SETTINGS.backtest_split_date
    df, rep = load_records()
    sold = sold_stones(df, drop_outliers=True)
    train, test, info = time_split(sold, split_date)

    eng = PricingEngine(EngineConfig(split_date=split_date)).fit(train)
    sugg = eng.predict(test)
    pred = np.array([s.suggested_discount for s in sugg])
    lo = np.array([s.ci_discount_low for s in sugg])
    hi = np.array([s.ci_discount_high for s in sugg])
    actual = test["FDiscount"].to_numpy()

    base = HierarchicalMedianModel().fit(train).predict(test)

    return {
        "report": rep.summary(),
        "split": info.__dict__,
        "engine": M.compute(pred, test).as_dict(),
        "baseline": M.compute(base, test).as_dict(),
        "signed_bias": round(float(np.mean(pred - actual)), 2),
        "coverage_target": eng.cfg.coverage,
        "coverage_empirical": round(M.interval_calibration(lo, hi, actual), 3),
        "median_band_width": round(float(np.median(hi - lo)), 2),
        "methods": dict(Counter(s.method for s in sugg)),
        "flags": dict(Counter(f for s in sugg for f in s.flags)),
        "test_df": test, "pred": pred, "base": base,
    }


def main() -> None:
    r = run()
    e, b = r["engine"], r["baseline"]
    print("=" * 78)
    print("GLOW STAR PRICING ENGINE — FINAL OUT-OF-TIME BACKTEST")
    print("=" * 78)
    print(r["report"])
    s = r["split"]
    print(f"\nSplit @ {s['split_date']}:  train={s['n_train']:,}  test={s['n_test']:,}  "
          "(leakage-free; recency-weighted; market-anchored; conformal bands)\n")
    print(f"  ENGINE    MAE={e['mae']:.2f}  MedAE={e['medae']:.2f}  "
          f"±3={e['within3']:.1%}  ±5={e['within5']:.1%}  "
          f"$MedAE={e['dollar_medae']:,.0f}  $MAE={e['dollar_mae']:,.0f}")
    print(f"  BASELINE  MAE={b['mae']:.2f}  MedAE={b['medae']:.2f}  "
          f"±3={b['within3']:.1%}  ±5={b['within5']:.1%}  "
          f"$MedAE={b['dollar_medae']:,.0f}  $MAE={b['dollar_mae']:,.0f}")
    lift = b["mae"] - e["mae"]
    print(f"\n  -> engine beats baseline by {lift:.2f} MAE pts ({lift / b['mae']:.1%} lower); "
          f"signed bias {r['signed_bias']:+.2f}")
    print(f"  CONFIDENCE  target={r['coverage_target']:.0%}  "
          f"empirical={r['coverage_empirical']:.1%}  "
          f"median width={r['median_band_width']:.1f} pts")
    print(f"  METHODS  {r['methods']}")
    print(f"  FLAGS    {r['flags']}")

    # Per-shape.
    test, pred, base = r["test_df"], r["pred"], r["base"]
    rows = []
    for shape, idx in test.groupby("Shape_full", observed=True).groups.items():
        sub = test.loc[idx]
        if len(sub) < 25:
            continue
        sel = test.index.get_indexer(idx)
        rows.append({"shape": shape, "n": len(sub),
                     "engine_mae": round(M.compute(pred[sel], sub).mae, 2),
                     "base_mae": round(M.compute(base[sel], sub).mae, 2),
                     "engine_within5": round(M.compute(pred[sel], sub).within5, 3)})
    tbl = pd.DataFrame(rows).sort_values("n", ascending=False)
    print("\nPER-SHAPE:")
    print(tbl.to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
