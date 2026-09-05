"""Production accuracy from what was ACTUALLY served.

Joins every quote the API recorded (store: quotes table) to the stone's later
realized sale in records.json, and measures variance = quoted - realized.
This is the only number that reflects the pipeline the client received, with the
grid/model/Rap that were live at the moment of the quote.

Run ON THE SERVER (it reads the production DB via GS_DATABASE_URL in .env):

    cd /opt/glowstar && sudo -u glowstar bash -c 'set -a; . ./.env; set +a; \
        .venv/bin/python scripts/server_quote_accuracy.py'

Read-only. Prints tables; writes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glowstar.store.db import get_engine, database_url  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def _summ(g: pd.DataFrame) -> pd.Series:
    a = g["abs"]
    return pd.Series({
        "n": len(g), "MAE": a.mean(), "MedAE": a.median(), "bias": g["err"].mean(),
        "within2": (a <= 2).mean(), "within5": (a <= 5).mean(), "ge5": (a >= 5).mean(),
        "cov": g["cov"].mean(), "width": g["width"].mean(),
    })


def main() -> None:
    print("DB:", database_url().split("@")[-1] if "@" in database_url() else database_url())
    eng = get_engine()
    q = pd.read_sql("select * from quotes", eng)
    d = pd.read_sql("select * from decisions", eng)
    print(f"quotes {len(q)}  decisions {len(d)}")
    q["ts"] = pd.to_datetime(q["ts"], utc=True, errors="coerce")
    q = q.dropna(subset=["ts", "discount"])
    q = q[~q["source"].isin(["perf", "test"])]
    q = q[~q["model_version"].isin(["v", "test", None])]

    print("\n=== served model version by week (a version that lingers = API not reloading) ===")
    q["week"] = q["ts"].dt.to_period("W").astype(str)
    print(q.pivot_table(index="week", columns="model_version", values="id", aggfunc="count", fill_value=0).to_string())

    print("\n=== quotes per source ===")
    print(q["source"].value_counts().to_string())

    print("\n=== flag frequency on served quotes ===")
    fl = q["flags"].dropna().map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else [])
    print(fl.explode().value_counts().head(25).to_string())
    q["no_cell"] = fl.map(lambda L: "no_grid_cell" in L).reindex(q.index).fillna(False)
    q["fallback"] = q["method"].eq("fallback")

    print("\n=== shape spelling stored on quotes (raw codes here = audit trail bug) ===")
    print(q["shape"].value_counts().head(25).to_string())

    rec_path = ROOT / "records.json"
    r = json.load(open(rec_path, encoding="utf-8"))
    recs = r if isinstance(r, list) else (r.get("records") or r.get("data"))
    R = pd.DataFrame(recs)
    print(f"\nrecords.json rows {len(R)}  status {R['Status'].value_counts().to_dict()}")
    sold = R[R["Status"].eq("Sold")].copy()
    sold["OrderDate"] = pd.to_datetime(sold["OrderDate"], utc=True, errors="coerce")
    sold = sold.dropna(subset=["OrderDate", "FDiscount"])
    sold["FDiscount"] = pd.to_numeric(sold["FDiscount"], errors="coerce")

    # Join on stone_id first, certificate second.
    j1 = q.merge(sold[["StoneId", "CertificateNo", "OrderDate", "FDiscount", "Shape_full", "Weight", "Color",
                       "Clarity", "CPS", "Fluorescence", "Lab", "Rap"]],
                 left_on="stone_id", right_on="StoneId", how="inner")
    j2 = q[q["certificate_no"].notna()].merge(
        sold[["StoneId", "CertificateNo", "OrderDate", "FDiscount", "Shape_full", "Weight", "Color",
              "Clarity", "CPS", "Fluorescence", "Lab", "Rap"]],
        left_on="certificate_no", right_on="CertificateNo", how="inner")
    j = pd.concat([j1, j2]).drop_duplicates(subset=["id"])
    j = j[j["OrderDate"] >= j["ts"].dt.normalize()]          # sale AFTER the quote
    j["lead_days"] = (j["OrderDate"] - j["ts"]).dt.days
    # keep the LAST quote before each sale
    j = j.sort_values("ts").groupby(["StoneId", "OrderDate"]).tail(1)
    print(f"\nquotes matched to a LATER realized sale: {len(j)}  "
          f"(distinct stones {j['StoneId'].nunique()}), median lead {j['lead_days'].median():.0f} days")
    if len(j) == 0:
        print("No matches - stone_id/certificate naming differs between quotes and records. Stop here.")
        return

    j["err"] = j["discount"] - j["FDiscount"]
    j["abs"] = j["err"].abs()
    j["cov"] = (j["FDiscount"] >= j["ci_low"]) & (j["FDiscount"] <= j["ci_high"])
    j["width"] = j["ci_high"] - j["ci_low"]
    j["round"] = j["Shape_full"].eq("Round")
    j["rap_match"] = np.isclose(j["rap"].astype(float), j["Rap"].astype(float), rtol=0.002)

    print("\n=== PRODUCTION ACCURACY: quoted vs realized (signed err = quoted - realized; + = we were too shallow/expensive) ===")
    print(_summ(j).round(3).to_string())
    print("\n--- by round/fancy x no_cell ---")
    print(j.groupby(["round", "no_cell"]).apply(_summ).round(3).to_string())
    print("\n--- by method ---")
    print(j.groupby("method").apply(_summ).round(3).to_string())
    print("\n--- by source ---")
    print(j.groupby("source").apply(_summ).round(3).to_string())
    print("\n--- by model version ---")
    print(j.groupby("model_version").apply(_summ).round(3).to_string())
    print("\n--- by shape (n>=15) ---")
    s = j.groupby("Shape_full").apply(_summ)
    print(s[s["n"] >= 15].sort_values("MAE", ascending=False).round(3).to_string())
    edges = [0, .30, .40, .50, .70, .90, 1.0, 1.5, 2.0, 3.0, 99]
    j["band"] = pd.cut(j["Weight"].astype(float), edges, right=False)
    print("\n--- by size band ---")
    print(j.groupby("band", observed=True).apply(_summ).round(3).to_string())
    print("\n--- by lead time (days between quote and sale) ---")
    j["lead_bkt"] = pd.cut(j["lead_days"], [-1, 3, 7, 14, 30, 60, 9999])
    print(j.groupby("lead_bkt", observed=True).apply(_summ).round(3).to_string())
    print("\n--- by fluorescence ---")
    print(j.groupby("Fluorescence").apply(_summ).round(3).to_string())
    print("\n--- by lab ---")
    print(j.groupby("Lab").apply(_summ).round(3).to_string())
    print("\n--- Rap on quote == client's Rap at sale? (False = stale Rap sheet band) ---")
    print(j.groupby(["band", "rap_match"], observed=True).size().unstack(fill_value=0).to_string())

    print("\n--- worst 25 ---")
    cols = ["ts", "stone_id", "Shape_full", "Weight", "Color", "Clarity", "CPS", "Fluorescence", "discount",
            "FDiscount", "err", "method", "flags", "lead_days"]
    w = j.sort_values("abs", ascending=False).head(25)[cols].copy()
    w["ts"] = w["ts"].dt.strftime("%Y-%m-%d")
    print(w.round(2).to_string(index=False))

    # ---- Bracket-overlap check on the SERVER's own grid history -------------
    # For each matched quote, read the grid cell as of the quote date under the
    # shipped rule (first containing bracket) and under the narrowest-bracket rule.
    try:
        import bisect
        from glowstar.market.grid_history import GridHistory, canon_shape, _FLUOR
        gh = GridHistory.load()
        rows = []
        for _, r in j.iterrows():
            sh = canon_shape(r["Shape_full"])
            if sh is None:
                rows.append((np.nan, np.nan, False)); continue
            k = (sh, str(r["Color"]).upper(), str(r["Clarity"]).upper(), str(r["CPS"]).upper(),
                 _FLUOR.get(str(r["Fluorescence"] or "").upper(), "NON"))
            asof = r["ts"].strftime("%Y-%m-%d"); wt = float(r["Weight"])
            cands = []
            for lo, hi, dates, discs in gh._idx.get(k, []):
                if lo <= wt <= hi:
                    i = bisect.bisect_left(dates, asof)
                    cands.append((lo, hi, discs[i - 1] if i else None))
            valid = [c for c in cands if c[2] is not None]
            first = cands[0][2] if cands else None
            narrow = min(valid, key=lambda c: (c[1] - c[0], c[0]))[2] if valid else None
            conflict = bool(len(valid) > 1 and first is not None and narrow is not None and abs(first - narrow) >= 0.5)
            rows.append((first, narrow, conflict))
        j["grid_first"], j["grid_narrow"], j["bracket_conflict"] = zip(*rows)
        j["grid_first"] = pd.to_numeric(j["grid_first"]); j["grid_narrow"] = pd.to_numeric(j["grid_narrow"])
        print("\n=== BRACKET OVERLAP on served quotes (grid as of quote date, server history) ===")
        print(f"quotes with a cell: {j['grid_first'].notna().sum()}   with a bracket conflict (>=0.5pt): {int(j['bracket_conflict'].sum())}")
        print("\n--- our quote vs sale, split by whether the stone's cell was conflicted ---")
        print(j.groupby("bracket_conflict").apply(_summ).round(3).to_string())
        c = j[j["bracket_conflict"]]
        if len(c):
            print("\n--- on conflicted stones: grid value vs SALE under each rule ---")
            print(f"first (shipped)   MAE {(c['grid_first'] - c['FDiscount']).abs().mean():.2f}")
            print(f"narrowest         MAE {(c['grid_narrow'] - c['FDiscount']).abs().mean():.2f}")
            print(f"our served quote  MAE {c['abs'].mean():.2f}")
            print("share of ALL >=5pt misses that are conflicted stones: "
                  f"{(j[j['abs'] >= 5]['bracket_conflict'].mean() * 100):.0f}%")
    except Exception as e:
        print("bracket check skipped:", e)

    if len(d):
        d = d.dropna(subset=["suggested_discount", "human_discount"])
        d["var"] = d["human_discount"] - d["suggested_discount"]
        print(f"\n=== desk decisions with a price: {len(d)}  ===")
        print(d["decision"].value_counts().to_string())
        print("desk moves us (human - ours): median %.2f  mean %.2f  |abs| mean %.2f  share deeper %.0f%%" % (
            d["var"].median(), d["var"].mean(), d["var"].abs().mean(), (d["var"] < 0).mean() * 100))
        print("exactly 0.0 variance count (echo red flag if large):", int((d["var"].abs() < 1e-9).sum()))


if __name__ == "__main__":
    main()
