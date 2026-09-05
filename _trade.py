import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from glowstar.service.tradeability import tradeability_for, build_table, _segment
from glowstar.service.pricing_service import PricingService, StoneIn
from glowstar.data.loaders import load_records, stock_stones

t = build_table(force=True)
print(f"tradeability table: {len(t.by_segment)} segments, cutoffs(days)={tuple(round(c) for c in t.cutoffs)}")
svc = PricingService()

print("\n=== SAME segment, DIFFERENT weight: label vs price ===")
print("(label is shape|colour|clarity only — weight is ignored)\n")
for shape, col, cla in [("Round","G","VS1"), ("Round","F","VS2"), ("Oval","G","VS1"), ("Pear","H","SI1")]:
    lab = None
    print(f"--- {shape} {col}/{cla} ---")
    for w in (0.31, 0.51, 0.71, 1.01, 1.51, 2.01, 3.01):
        tr = tradeability_for(shape, w, col, cla)
        try:
            r = svc.price(StoneIn(StoneId="X", Shape_full=shape, Weight=w, Color=col,
                                  Clarity=cla, CPS="3EX", Fluorescence="Non"), explain=False)["suggestion"]
            d, ppc = r["suggested_discount"], r["suggested_ppc"]
        except Exception as e:
            d, ppc = float("nan"), float("nan")
        print(f"   {w:>5}ct  Tradeability={str(tr['label']):<10} ({tr['median_days']}d)   "
              f"price={d:+7.2f}  (${ppc:,.0f}/ct)")
    print()

print("=== does the label track the price at all? (live stock) ===")
df,_ = load_records(); st = stock_stones(df)
rows=[]
for r in st.sample(min(1200, len(st)), random_state=0).itertuples():
    tr = tradeability_for(r.Shape_full, r.Weight, r.Color, r.Clarity)
    if tr["label"]:
        rows.append(dict(label=tr["label"], days=tr["median_days"], w=float(r.Weight)))
d = pd.DataFrame(rows)
print(d.groupby("label").agg(n=("w","size"), median_ct=("w","median"),
      min_ct=("w","min"), max_ct=("w","max"), days=("days","median")).to_string())
print("\n=> one label spans the entire weight range. A 0.3ct and a 3ct share it.")
