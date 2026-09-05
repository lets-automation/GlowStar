"""Paired A/B of grid-lookup rules on THIS machine's data: same stones, same weekly
origins, shipped serving_config(), grid joined point-in-time per row.

Arms:
  old       first containing bracket (the rule that shipped before 2026-09-05)
  new       narrowest containing bracket (shipped 2026-09-05)
  new_min30 narrowest bracket, but NO grid cell below 0.30 ct. Why: measured on the
            server 2026-09-05, `new` improved overall MAE (-0.10) but the <0.30 ct
            segment went 2.84 -> 4.25 and its >=5pt tail 13% -> 39%. Below 0.30 ct
            the grid is 14-21 pts from the sale under EVERY rule (the desk does not
            maintain those cells); once the 0.30+ grid became trustworthy the refit
            model leaned on the grid harder and that trust hurt exactly there.

The nightly gate cannot answer this: it scores tonight's candidate on tonight's
7-day window against yesterday's number on yesterday's window, and that
composition noise is +/-0.05-0.1 a night. This runs every arm on identical folds.

Run ON THE SERVER (about 12-15 minutes for 3 origins x 3 arms; read-only):

    cd /opt/glowstar && sudo -u glowstar bash -c 'set -a; . ./.env; set +a; \
        .venv/bin/python scripts/server_bracket_ab.py 3' 2>&1 | grep -v "fluoro caps" | tee /tmp/bracket_ab_$(date +%F).txt

Optional: `... server_bracket_ab.py 3 old,new_min30` to run only some arms.
"""
from __future__ import annotations

import bisect
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import glowstar.market.grid_history as GH
from glowstar.data.loaders import load_records, sold_stones
from glowstar.models.engine import PricingEngine
from glowstar.training.retrain import serving_config

N_ORIGINS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ARMS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["old", "new", "new_min30", "new_clean"]
HORIZON = int(sys.argv[3]) if len(sys.argv) > 3 else 7      # days per test window
# e.g. `1 old,new 3` -> ONE origin three days before the last sale: train on
# everything before it, price the last three days. Use it to ask "does having the
# first days of a batch in training fix the rest of the batch?"
MIN_GRID_WEIGHT = 0.30
pd.set_option("display.width", 220)

NEW_READ_KEY = GH.GridHistory._read_key
NEW_AS_OF_DETAILED = GH.GridHistory.as_of_detailed


def old_read_key(self, key, weight, asof):
    """Pre-2026-09-05 rule: first containing bracket; unedited first bracket -> None."""
    for lo, hi, dates, discs in self._idx.get(key, []):
        if lo <= weight <= hi:
            i = bisect.bisect_left(dates, asof)
            if i == 0:
                return None, None
            age = None
            try:
                age = (datetime.strptime(asof, "%Y-%m-%d") - datetime.strptime(dates[i - 1], "%Y-%m-%d")).days
            except ValueError:
                pass
            return discs[i - 1], age
    return None, None


def as_of_detailed_min30(self, shape, weight, color, clarity, cps, fluorescence, asof):
    try:
        if float(weight) < MIN_GRID_WEIGHT:
            return None, None, None
    except (TypeError, ValueError):
        pass
    return NEW_AS_OF_DETAILED(self, shape, weight, color, clarity, cps, fluorescence, asof)


# Plausible cell values. The grid store holds 608 cells whose latest value is
# exactly 0.0 (517 of them in 0.30-0.99 ct) and 1,187 versions below -95: unfilled
# placeholders and impossible prices. A narrow placeholder would beat a real coarse
# cell under "narrowest wins", so such brackets are skipped (next-narrowest answers).
PLAUSIBLE_LO, PLAUSIBLE_HI = -95.0, 30.0


def _plausible(v):
    return v != 0.0 and PLAUSIBLE_LO <= v <= PLAUSIBLE_HI


def clean_read_key(self, key, weight, asof):
    """Narrowest containing bracket whose as-of value is plausible."""
    best = None
    for lo, hi, dates, discs in self._idx.get(key, []):
        if lo <= weight <= hi:
            i = bisect.bisect_left(dates, asof)
            if i == 0 or not _plausible(discs[i - 1]):
                continue
            cand = (hi - lo, lo, discs[i - 1], dates[i - 1])
            if best is None or cand[:2] < best[:2]:
                best = cand
    if best is None:
        return None, None
    age = None
    try:
        age = (datetime.strptime(asof, "%Y-%m-%d") - datetime.strptime(best[3], "%Y-%m-%d")).days
    except ValueError:
        pass
    return best[2], age


ARM_DEFS = {
    "old":       (old_read_key,   NEW_AS_OF_DETAILED),
    "new":       (NEW_READ_KEY,   NEW_AS_OF_DETAILED),
    "new_min30": (NEW_READ_KEY,   as_of_detailed_min30),
    "new_clean": (clean_read_key, as_of_detailed_min30),   # min30 + plausibility filter
}


def apply_arm(name):
    rk, aod = ARM_DEFS[name]
    GH.GridHistory._read_key = rk
    GH.GridHistory.as_of_detailed = aod


def restore():
    GH.GridHistory._read_key = NEW_READ_KEY
    GH.GridHistory.as_of_detailed = NEW_AS_OF_DETAILED


def run_arm(name, train, test):
    apply_arm(name)
    t0 = time.time()
    eng = PricingEngine(serving_config()).fit(train)
    te = GH.attach_grid(test, eng.grid_history, asof=None) if getattr(eng, "grid_history", None) is not None else test
    sug = eng.predict(te)
    out = pd.DataFrame({
        "stone": [s.stone_id for s in sug],
        f"pred_{name}": [s.suggested_discount for s in sug],
        f"lo_{name}": [s.ci_discount_low for s in sug],
        f"hi_{name}": [s.ci_discount_high for s in sug],
        f"grid_{name}": te["grid_discount"].to_numpy() if "grid_discount" in te.columns else np.nan,
    })
    print(f"    {name:10s} fit+predict {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    df, _ = load_records()
    sold_all = sold_stones(df, drop_outliers=False).dropna(subset=["OrderDate_dt"]).sort_values("OrderDate_dt")
    sold_clean = sold_stones(df, drop_outliers=True)
    end = sold_all["OrderDate_dt"].max().normalize()
    print(f"records {len(df)}  sold {len(sold_all)}  last sale {end.date()}  origins {N_ORIGINS}  arms {ARMS}", flush=True)
    rows = []
    for k in range(N_ORIGINS, 0, -1):
        origin = end - pd.Timedelta(days=HORIZON * k - 1)
        train = sold_clean[sold_clean["OrderDate_dt"] < origin]
        test = sold_all[(sold_all["OrderDate_dt"] >= origin) & (sold_all["OrderDate_dt"] < origin + pd.Timedelta(days=HORIZON))]
        if test.empty:
            continue
        print(f"origin {origin.date()}  train {len(train)}  test {len(test)}", flush=True)
        m = None
        for arm in ARMS:
            a = run_arm(arm, train, test)
            m = a if m is None else m.merge(a, on="stone")
        base = test[["StoneId", "FDiscount", "Shape_full", "Weight"]].rename(columns={"StoneId": "stone"})
        m = m.merge(base, on="stone")
        m["origin"] = origin.date()
        rows.append(m)
    restore()
    m = pd.concat(rows, ignore_index=True)
    m["act"] = m["FDiscount"].astype(float)
    for arm in ARMS:
        m[f"abs_{arm}"] = (m[f"pred_{arm}"] - m["act"]).abs()
        m[f"cov_{arm}"] = (m["act"] >= m[f"lo_{arm}"]) & (m["act"] <= m[f"hi_{arm}"])
    m["round"] = m["Shape_full"].eq("Round")
    m["small"] = m["Weight"].astype(float) < MIN_GRID_WEIGHT
    ref = ARMS[0]

    def s(g):
        d = {"n": len(g)}
        for arm in ARMS:
            d[f"mae_{arm}"] = g[f"abs_{arm}"].mean()
        for arm in ARMS:
            d[f"ge5_{arm}"] = (g[f"abs_{arm}"] >= 5).mean()
        for arm in ARMS:
            d[f"cov_{arm}"] = g[f"cov_{arm}"].mean()
        return pd.Series(d)

    print(f"\n=== PAIRED, ALL STONES (reference arm: {ref}) ===")
    print(s(m).round(3).to_string())
    print("\n=== by small (<0.30ct) x round ===")
    print(m.groupby(["small", "round"]).apply(s).round(3).to_string())
    print("\n=== by origin ===")
    print(m.groupby("origin").apply(s).round(3).to_string())
    edges = [0, .3, .4, .5, .7, .9, 1.0, 1.5, 2.0, 3.0, 99]
    m["band"] = pd.cut(m["Weight"].astype(float), edges, right=False)
    print("\n=== by size band ===")
    print(m.groupby("band", observed=True).apply(s).round(3).to_string())

    rng = np.random.default_rng(0)
    print(f"\n=== paired delta vs '{ref}' (negative = better than {ref}) ===")
    for arm in ARMS[1:]:
        d = (m[f"abs_{arm}"] - m[f"abs_{ref}"]).to_numpy()
        bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        lo_, hi_ = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        tail = (m[f"abs_{arm}"] >= 5).mean() - (m[f"abs_{ref}"] >= 5).mean()
        verdict = "BETTER" if hi_ < 0 else ("WORSE" if lo_ > 0 else "NOT DISTINGUISHABLE")
        print(f"{arm:10s} dMAE {d.mean():+.4f} [{lo_:+.4f}, {hi_:+.4f}]  d(>=5pt tail) {tail*100:+.2f} pts  -> {verdict}"
              + ("  (but tail WORSE)" if verdict == "BETTER" and tail > 0.002 else ""))
    try:
        import tempfile
        out = Path(tempfile.gettempdir()) / "bracket_ab_rows.csv"
        m.to_csv(out, index=False)
        print("per-stone rows:", out)
    except Exception as e:
        print("per-stone CSV not written:", e)
    print("DONE")


if __name__ == "__main__":
    main()
