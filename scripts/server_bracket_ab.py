"""Paired A/B of the grid bracket rule on THIS machine's data: old rule (first
containing bracket) vs new rule (narrowest containing bracket), same stones, same
weekly origins, shipped serving_config(), grid joined point-in-time per row.

The nightly gate cannot answer this: it scores tonight's candidate on tonight's
7-day window against yesterday's number on yesterday's window, and that
composition noise is +/-0.05-0.1 a night. This runs both rules on identical folds.

Run ON THE SERVER (about 10-15 minutes for 3 origins; read-only, writes only /tmp):

    cd /opt/glowstar && sudo -u glowstar bash -c 'set -a; . ./.env; set +a; \
        .venv/bin/python scripts/server_bracket_ab.py 3' 2>&1 | tee /tmp/bracket_ab_$(date +%F).txt
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
HORIZON = 7
pd.set_option("display.width", 200)

NEW_READ_KEY = GH.GridHistory._read_key


def old_read_key(self, key, weight, asof):
    """The rule that shipped before 2026-09-05: first containing bracket decides,
    and an unedited first bracket returns None."""
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


def run_arm(name, read_key, train, test):
    GH.GridHistory._read_key = read_key
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
    print(f"    {name}: fit+predict {time.time() - t0:.0f}s", flush=True)
    return out


def main():
    df, _ = load_records()
    sold_all = sold_stones(df, drop_outliers=False).dropna(subset=["OrderDate_dt"]).sort_values("OrderDate_dt")
    sold_clean = sold_stones(df, drop_outliers=True)
    end = sold_all["OrderDate_dt"].max().normalize()
    print(f"records {len(df)}  sold {len(sold_all)}  last sale {end.date()}  origins {N_ORIGINS}", flush=True)
    rows = []
    for k in range(N_ORIGINS, 0, -1):
        origin = end - pd.Timedelta(days=HORIZON * k - 1)
        train = sold_clean[sold_clean["OrderDate_dt"] < origin]
        test = sold_all[(sold_all["OrderDate_dt"] >= origin) & (sold_all["OrderDate_dt"] < origin + pd.Timedelta(days=HORIZON))]
        if test.empty:
            continue
        print(f"origin {origin.date()}  train {len(train)}  test {len(test)}", flush=True)
        a = run_arm("old", old_read_key, train, test)
        b = run_arm("new", NEW_READ_KEY, train, test)
        m = a.merge(b, on="stone")
        base = test[["StoneId", "FDiscount", "Shape_full", "Weight"]].rename(columns={"StoneId": "stone"})
        m = m.merge(base, on="stone")
        m["origin"] = origin.date()
        rows.append(m)
    GH.GridHistory._read_key = NEW_READ_KEY
    m = pd.concat(rows, ignore_index=True)
    m["act"] = m["FDiscount"].astype(float)
    m["abs_old"] = (m["pred_old"] - m["act"]).abs()
    m["abs_new"] = (m["pred_new"] - m["act"]).abs()
    m["cov_old"] = (m["act"] >= m["lo_old"]) & (m["act"] <= m["hi_old"])
    m["cov_new"] = (m["act"] >= m["lo_new"]) & (m["act"] <= m["hi_new"])
    m["round"] = m["Shape_full"].eq("Round")
    m["grid_changed"] = ~np.isclose(m["grid_old"].fillna(-999), m["grid_new"].fillna(-999))

    def s(g):
        return pd.Series(dict(n=len(g), old=g["abs_old"].mean(), new=g["abs_new"].mean(),
                              delta=(g["abs_new"] - g["abs_old"]).mean(),
                              ge5_old=(g["abs_old"] >= 5).mean(), ge5_new=(g["abs_new"] >= 5).mean(),
                              w2_old=(g["abs_old"] <= 2).mean(), w2_new=(g["abs_new"] <= 2).mean(),
                              cov_old=g["cov_old"].mean(), cov_new=g["cov_new"].mean(),
                              improved=(g["abs_new"] < g["abs_old"] - 0.05).mean(),
                              worsened=(g["abs_new"] > g["abs_old"] + 0.05).mean()))

    print("\n=== PAIRED, ALL STONES (signed delta < 0 means the new rule is better) ===")
    print(s(m).round(3).to_string())
    print("\n=== by round x whether the stone's grid cell changed ===")
    print(m.groupby(["round", "grid_changed"]).apply(s).round(3).to_string())
    print("\n=== by origin ===")
    print(m.groupby("origin").apply(s)[["n", "old", "new", "delta", "ge5_old", "ge5_new"]].round(3).to_string())
    edges = [0, .3, .4, .5, .7, .9, 1.0, 1.5, 2.0, 3.0, 99]
    m["band"] = pd.cut(m["Weight"].astype(float), edges, right=False)
    print("\n=== by size band ===")
    print(m.groupby("band", observed=True).apply(s)[["n", "old", "new", "delta", "ge5_old", "ge5_new"]].round(3).to_string())
    rng = np.random.default_rng(0)
    d = (m["abs_new"] - m["abs_old"]).to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    print(f"\ndelta MAE overall {d.mean():+.4f}   95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    print("VERDICT:", "NEW RULE BETTER" if np.percentile(bs, 97.5) < 0 else ("NEW RULE WORSE" if np.percentile(bs, 2.5) > 0 else "NOT DISTINGUISHABLE"))
    try:
        import tempfile
        out = Path(tempfile.gettempdir()) / "bracket_ab_rows.csv"
        m.to_csv(out, index=False)
        print("per-stone rows:", out)
    except Exception as e:                      # never lose the printed result over a file path
        print("per-stone CSV not written:", e)
    print("DONE")


if __name__ == "__main__":
    main()
