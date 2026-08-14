"""Is the 499 file actually ACCURATE? Two honest, out-of-time tests.

Verifying that a file is internally consistent (no nulls, bands ordered) says
nothing about whether the numbers are right. These two tests do.

TEST 1 — DISCOUNT. Train on sales before the split, then score stones sold
AFTER it that match this file's profile (same shapes, same size band, same
colour/clarity range). Nothing the model saw in training. Reports the error the
desk would actually have felt on stones like these.

TEST 2 — SPEED. Build the Fast/Medium/Slow table from an EARLIER window
(120-60 days ago), then check what those labels did in the LATER window
(last 60 days). If "Fast" segments really did sell faster than "Slow" ones, the
label carries signal. If they did not, it does not — and we should say so.
Scoring the labels on the same window used to build them would be circular.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from glowstar.config import SETTINGS
from glowstar.data.loaders import load_records, sold_stones
from glowstar.models.engine import PricingEngine
from glowstar.training.retrain import serving_config, _evaluate
from glowstar.validation.backtest import time_split
from glowstar.market.grid_history import attach_grid


def profile_mask(df: pd.DataFrame, src: pd.DataFrame) -> pd.Series:
    """Stones that look like the client's file."""
    shapes = {"Round", "Pear", "Marquise"}
    lo, hi = float(src["Carats"].min()), float(src["Carats"].max())
    cols = set(src["Color"].astype(str).str.strip())
    clar = set(src["Clarity"].astype(str).str.strip())
    return (df["Shape_full"].astype(str).isin(shapes)
            & df["Weight"].between(lo, hi)
            & df["Color"].astype(str).isin(cols)
            & df["Clarity"].astype(str).isin(clar))


def main() -> None:
    src = pd.read_excel("artifacts/499.xlsx")
    sold = sold_stones(load_records()[0], drop_outliers=True)

    # ---------------- TEST 1: discount accuracy, out-of-time ----------------
    train, test, _ = time_split(sold, SETTINGS.backtest_split_date)
    eng = PricingEngine(serving_config(SETTINGS.backtest_split_date)).fit(train)

    m = profile_mask(test, src)
    like = test[m]
    print(f"\n=== TEST 1 — DISCOUNT (out-of-time, stones like this file) ===")
    print(f"held-out stones matching the file's profile: {len(like):,}")
    if len(like) < 50:
        print("too few to judge — reporting the whole window instead")
        like = test

    like = attach_grid(like, eng.grid_history)      # point-in-time, honest
    r = _evaluate(eng, like)
    print(f"  MAE          {r['mae']}")
    print(f"  within +-2   {r.get('within2')}   <- the metric the desk grades on")
    print(f"  within +-5   {r['within5']}")
    print(f"  bias         {r['bias']}")
    print(f"  band holds   {r['coverage']}  (we publish 80%)")

    # baseline for context: what a naive segment-median would have scored
    seg = (like["Shape_full"].astype(str) + "|" + like["Color"].astype(str)
           + "|" + like["Clarity"].astype(str))
    med = train.assign(seg=(train["Shape_full"].astype(str) + "|"
                            + train["Color"].astype(str) + "|"
                            + train["Clarity"].astype(str))
                       ).groupby("seg")["FDiscount"].median()
    base = seg.map(med).fillna(train["FDiscount"].median())
    print(f"  naive segment-median baseline MAE: "
          f"{np.abs(base.to_numpy() - like['FDiscount'].to_numpy()).mean():.3f}")

    # ---------------- TEST 2: do the speed labels predict? ------------------
    df, _ = load_records()
    now = pd.Timestamp.now().normalize()
    s = df[df["Status"] == "Sold"].copy()
    s["dur"] = (s["OrderDate_dt"] - s["CreatedDate_dt"]).dt.days
    s = s[s["dur"].notna() & (s["dur"] >= 0)]
    s["seg"] = (s["Shape_full"].astype(str) + "|" + s["Color"].astype(str)
                + "|" + s["Clarity"].astype(str))

    earlier = s[(s["OrderDate_dt"] < now - pd.Timedelta(days=60))
                & (s["OrderDate_dt"] >= now - pd.Timedelta(days=120))]
    later = s[s["OrderDate_dt"] >= now - pd.Timedelta(days=60)]

    tab = earlier.groupby("seg")["dur"].agg(["median", "size"])
    tab = tab[tab["size"] >= 15]["median"]
    c1, c2 = tab.quantile(1 / 3), tab.quantile(2 / 3)

    def lab(v):
        return "Fast" if v <= c1 else ("Medium" if v <= c2 else "Slow")

    later = later[later["seg"].isin(tab.index)].copy()
    later["label"] = later["seg"].map(tab).map(lab)

    print(f"\n=== TEST 2 — SPEED (labels built on days 120-60, tested on last 60) ===")
    print(f"segments labelled: {len(tab)}   stones tested: {len(later):,}")
    g = later.groupby("label")["dur"].agg(["median", "mean", "size"])
    print(g.reindex(["Fast", "Medium", "Slow"]).to_string())
    if {"Fast", "Slow"} <= set(g.index):
        print(f"\n  Fast median {g.loc['Fast','median']:.0f}d vs Slow median "
              f"{g.loc['Slow','median']:.0f}d  -> separation "
              f"{g.loc['Slow','median'] - g.loc['Fast','median']:+.0f} days")


if __name__ == "__main__":
    main()
