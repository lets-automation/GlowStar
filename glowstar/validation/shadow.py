"""Shadow mode: how well the engine reproduces the human pricer (brief Sec 12).

IMPORTANT construct (handled honestly): on SOLD stones the human's listed
`Discount` is ~equal to the realized `FDiscount` (the human set the price that
became the outcome), so a "human-vs-realized accuracy" contest is rigged — the
human always wins by construction. That is NOT what shadow mode measures.

Shadow mode measures **agreement**: pricing from stone ATTRIBUTES ONLY (no
leakage, no sight of the human's decision), how often does the engine land near
the expert's decision? High agreement -> the engine has learned the desk's
judgment and can carry routine pricing at scale/speed. DIVERGENCES are the
valuable output: stones to review (either the engine or the human is off), and
the queue for the human-feedback loop. True realized-outcome superiority needs
unsold/forward data and is measured later as snapshots accrue.

Go-live is gated per segment on agreement: recommend only where the engine
matches the human within +/-5 pts on a high share of stones.

Run:  python -m glowstar.validation.shadow
"""

from __future__ import annotations

import numpy as np

from ..config import SETTINGS
from ..data.loaders import load_records, sold_stones
from ..models.engine import PricingEngine, EngineConfig
from .backtest import time_split


def run(split_date: str | None = None, agree_within: float = 5.0,
        go_live_share: float = 0.60, min_n: int = 25) -> dict:
    split_date = split_date or SETTINGS.backtest_split_date
    df, _ = load_records()
    sold = sold_stones(df, drop_outliers=True)
    train, test, info = time_split(sold, split_date)

    eng = PricingEngine(EngineConfig(split_date=split_date)).fit(train)
    engine_pred = np.array([s.suggested_discount for s in eng.predict(test)])
    human = test["Discount"].to_numpy()           # the pricer's decision
    realized = test["FDiscount"].to_numpy()

    gap = engine_pred - human                      # + = engine prices higher (shallower)
    agree = np.abs(gap)
    diverge = agree > 10.0                          # material disagreement -> review

    def seg_rows(by: str):
        rows = []
        for seg, idx in test.groupby(by, observed=True).groups.items():
            sel = test.index.get_indexer(idx)
            if len(sel) < min_n:
                continue
            a = agree[sel]
            within = float(np.mean(a <= agree_within))
            rows.append({
                "segment": seg, "n": len(sel),
                "agree_mae": round(float(a.mean()), 2),
                "agree_within5": round(within, 3),
                "mean_gap": round(float(gap[sel].mean()), 2),
                "diverge_n": int(diverge[sel].sum()),
                "go_live": bool(within >= go_live_share),
            })
        return sorted(rows, key=lambda r: r["n"], reverse=True)

    return {
        "split": info.__dict__,
        "overall": {
            "agree_mae": round(float(agree.mean()), 2),
            "agree_within5": round(float(np.mean(agree <= agree_within)), 3),
            "mean_gap": round(float(gap.mean()), 2),
            "diverge_count": int(diverge.sum()),
            "diverge_share": round(float(diverge.mean()), 3),
            "note": "agreement with the human decision; divergences are review cases.",
        },
        "by_shape": seg_rows("Shape_full"),
    }


def main() -> None:
    r = run()
    o = r["overall"]
    print("=" * 74)
    print("SHADOW MODE -- engine agreement with human pricers (attributes only)")
    print("=" * 74)
    print(f"Overall: agree MAE={o['agree_mae']}  within +/-5 = {o['agree_within5']:.0%}  "
          f"mean gap={o['mean_gap']:+.2f} pts")
    print(f"Material divergences (>10 pts): {o['diverge_count']} "
          f"({o['diverge_share']:.0%}) -> human review queue\n")
    print(f"{'segment':<14}{'n':>6}{'agree_mae':>11}{'within5':>9}{'mean_gap':>10}{'review':>8}  go-live")
    for row in r["by_shape"]:
        print(f"{row['segment']:<14}{row['n']:>6}{row['agree_mae']:>11}{row['agree_within5']:>9.0%}"
              f"{row['mean_gap']:>10.2f}{row['diverge_n']:>8}   {'YES' if row['go_live'] else 'hold'}")
    live = [r2["segment"] for r2 in r["by_shape"] if r2["go_live"]]
    print(f"\nRecommended go-live (high agreement): {live or 'none yet'}")
    print("Elsewhere: engine assists, human decides; divergences feed the feedback loop.")
    print("=" * 74)


if __name__ == "__main__":
    main()
