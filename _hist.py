"""Does the missing Dec-Jan history cost accuracy?
FULL (29,947 sold, dev) vs TRUNCATED (drop pre-2026-02-01, ~= the server's 21,572).
Both scored on the SAME rolling 7-day production-horizon windows."""
import warnings, time, numpy as np, pandas as pd, logging
warnings.filterwarnings("ignore"); logging.basicConfig(level=logging.ERROR)
from glowstar.data.loaders import load_records, sold_stones
from glowstar.models.engine import PricingEngine
from glowstar.training.retrain import serving_config
from glowstar.market.grid_history import attach_grid

sold = sold_stones(load_records()[0], drop_outliers=True)
sold = sold[pd.to_numeric(sold["FDiscount"], errors="coerce").notna()]
CUT = pd.Timestamp("2026-02-01")
print(f"FULL sold={len(sold):,}   TRUNCATED sold={int((sold.OrderDate_dt>=CUT).sum()):,}", flush=True)
origins = pd.date_range("2026-07-13","2026-08-17",freq="7D")
res={}
for arm in ("FULL","TRUNCATED"):
    errs=[]
    for o in origins:
        tr = sold[sold.OrderDate_dt < o]
        if arm=="TRUNCATED": tr = tr[tr.OrderDate_dt >= CUT]
        te = sold[(sold.OrderDate_dt>=o)&(sold.OrderDate_dt<o+pd.Timedelta(days=7))]
        if len(te)<40: continue
        eng = PricingEngine(serving_config()).fit(tr)
        teg = attach_grid(te, eng.grid_history, asof=o)
        p = np.array([s.suggested_discount for s in eng.predict(teg, as_of=o)])
        errs.append(np.abs(p - te.FDiscount.to_numpy()))
    e=np.concatenate(errs); res[arm]=e
    print(f"{arm:<10} n_train_last={len(tr):>6}  MAE={e.mean():.4f}  w2={(e<=2).mean():.4f}  "
          f">=5pt={(e>=5).mean():.4f}", flush=True)
d = res["TRUNCATED"].mean()-res["FULL"].mean()
print(f"\ncost of losing Dec+Jan history: {d:+.4f} MAE  "
      f"({'FULL history is better' if d>0 else 'no benefit from the old data'})")
