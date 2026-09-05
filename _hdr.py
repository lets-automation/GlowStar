"""Round-only vs fancy accuracy on the PRODUCTION horizon — the exact numbers
the client message quotes. Trained strictly before the test window."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from glowstar.data.loaders import load_records, sold_stones
from glowstar.models.engine import PricingEngine
from glowstar.training.retrain import serving_config
import glowstar.market.grid_history as GH

df, _ = load_records()
sold = sold_stones(df, drop_outliers=True).dropna(subset=["OrderDate_dt"]).sort_values("OrderDate_dt")
end = sold.OrderDate_dt.max(); H = GH.GridHistory.load()
origin = end - pd.Timedelta(days=7)
tr, te_ = sold[sold.OrderDate_dt < origin], sold[sold.OrderDate_dt >= origin]
eng = PricingEngine(serving_config()).fit(tr)
te = GH.attach_grid(te_, H, asof=None)
pred = np.array([s.suggested_discount for s in eng.predict(te)], float)
act = te["FDiscount"].to_numpy(float)
e = np.abs(pred - act)
o = pd.DataFrame({"shape": te["Shape_full"].to_numpy(), "abs": e,
                  "cell": te["grid_discount"].notna().to_numpy()})
o["fancy"] = ~o["shape"].eq("Round")

def line(lbl, m):
    if m.sum() == 0: return
    x = o.loc[m, "abs"]
    print(f"  {lbl:34s} n={len(x):5d}  MAE={x.mean():5.2f}  "
          f"within2={(x<=2).mean()*100:5.1f}%  within5={(x<=5).mean()*100:5.1f}%")

print(f"PRODUCTION HORIZON (7d), train {len(tr)} / test {len(te)}")
line("ROUND (all)",            ~o.fancy)
line("ROUND with cell",        (~o.fancy) & o.cell)
line("FANCY with cell",        o.fancy & o.cell)
line("FANCY no cell",          o.fancy & ~o.cell)
line("ALL stones",             pd.Series(True, index=o.index))
print()
print(f"  share of test stones that are Round: {(~o.fancy).mean()*100:.1f}%")
print(f"  no-cell rate  Round {((~o.fancy)&~o.cell).sum()/max((~o.fancy).sum(),1)*100:.2f}%"
      f"   Fancy {(o.fancy&~o.cell).sum()/max(o.fancy.sum(),1)*100:.2f}%")
