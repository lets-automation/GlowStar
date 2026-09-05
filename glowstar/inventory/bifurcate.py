"""Five-class bifurcation and ageing buckets — Workstream B, Phase C.

This is the deliverable the client's north star names: *accurate pricing that
intelligently bifurcates inventory, so fast-moving and slow-moving stock are
priced with different strategies.* Everything here classifies; `reprice.py`
acts on the classification.

THE VOCABULARY IS THE MOU'S, AND THE MIGRATION IS NOT A FIND-AND-REPLACE
-------------------------------------------------------------------------
MOU 5.1 says **Fast / Semi-Fast / Medium / Semi-Slow / Slow** and those words are
canonical everywhere inside this system. But `service/tradeability.py` publishes
`High / Semi High / Medium / Semi Slow / Slow` on `/frontoffice/reason` today,
the client's FrontOffice spec asks for those words, and their screen is bound to
them right now. Rows already written to the `scores` table carry them too.

So there is exactly ONE mapping, here, in both directions:

  * everything new — models, endpoints, reports — emits the MOU words;
  * `to_frontoffice()` translates at the boundary, and only there;
  * `from_frontoffice()` translates history so a report that reads the `scores`
    table does not show the desk two vocabularies in one column, which is the
    precise confusion this decision exists to end.

Deleting the mapping is a coordinated change with the client's IT owner (MOU 2),
not a cleanup: MOU 9.1 requires written notice in both directions on a field
change, and renaming strings on a live endpoint for a cosmetic win breaks their
screen. Guarded by `test_frontoffice_vocabulary_is_unchanged`.

WHAT DECIDES A CLASS
--------------------
The velocity score — a percentile of expected days-to-sell against the client's
OWN book, so "Slow" means slow for Glow Star rather than slow against a number
invented here. For stock the estimate is CONDITIONAL on how long the stone has
already sat unsold, which is how "a stone past its segment median is slowing"
becomes arithmetic rather than a rule of thumb.

Ageing buckets (0-90 / 91-180 / 181-365 / 365+) are reported ALONGSIDE, never
folded in. They are absolute where the class is relative, and the pair is what
makes the classification legible: "Slow, and 200 days old" is a different
conversation from "Slow, listed last week".

Every row carries its basis — score, expected days and interval, age, the
segment it was judged in, how many sales stand behind that segment, and whether
it fell back to a coarser one (MOU 10.3).

Run:  python -m glowstar.inventory.bifurcate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# MOU 5.1 wording. Fastest first.
CLASSES: tuple[str, ...] = ("Fast", "Semi-Fast", "Medium", "Semi-Slow", "Slow")

# The LIVE FrontOffice vocabulary. One dict, one place, both directions — see
# the module docstring before changing either side.
FRONTOFFICE_LABELS: dict[str, str] = {
    "Fast": "High",
    "Semi-Fast": "Semi High",
    "Medium": "Medium",
    "Semi-Slow": "Semi Slow",
    "Slow": "Slow",
}
_FROM_FRONTOFFICE: dict[str, str] = {v: k for k, v in FRONTOFFICE_LABELS.items()}


def to_frontoffice(label: str | None) -> str | None:
    """MOU wording -> the words the client's screen reads today."""
    return None if label is None else FRONTOFFICE_LABELS.get(label, label)


def from_frontoffice(label: str | None) -> str | None:
    """The client's words (incl. rows already in the `scores` table) -> MOU wording."""
    return None if label is None else _FROM_FRONTOFFICE.get(label, label)


# Ageing buckets, MOU 5.1. Inclusive lower, inclusive upper.
AGEING_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0, 90, "0-90"),
    (91, 180, "91-180"),
    (181, 365, "181-365"),
    (366, float("inf"), "365+"),
)
RED_FLAG_BUCKET = "365+"


def ageing_bucket(age_days: float | None) -> str | None:
    if age_days is None or not np.isfinite(age_days):
        return None
    for lo, hi, name in AGEING_BUCKETS:
        if lo <= age_days <= hi:
            return name
    return AGEING_BUCKETS[-1][2]


@dataclass(frozen=True)
class BifurcationConfig:
    """Where the class boundaries sit, on the 0-100 own-velocity percentile.

    Even quintiles, so the five classes carve the client's own distribution the
    way `tradeability.py` already carves its own. That has a consequence worth
    stating rather than hiding, and the Legend sheet states it: the cutoffs are
    RELATIVE. A book that got uniformly faster would still have a slowest fifth.
    That is why the day cutoffs the percentiles imply are reported next to the
    classes, and why the absolute ageing buckets are reported beside them —
    between the three, a reader can always see whether "Slow" means 60 days or
    200.
    """

    cutoffs: tuple[int, int, int, int] = (80, 60, 40, 20)
    # A segment with fewer own sales than this does not get its own norm; the
    # basis says which coarser level answered instead.
    min_segment_sales: int = 15

    def label(self, score: float | None) -> str | None:
        if score is None or not np.isfinite(score):
            return None
        c1, c2, c3, c4 = self.cutoffs
        if score >= c1:
            return CLASSES[0]
        if score >= c2:
            return CLASSES[1]
        if score >= c3:
            return CLASSES[2]
        if score >= c4:
            return CLASSES[3]
        return CLASSES[4]


def serving_bifurcation_config() -> BifurcationConfig:
    """THE bifurcation config — same discipline as `retrain.serving_config()`.

    A knob that moves a stone between classes changes what the desk does with
    it, so it lives here or it does not ship.
    """
    return BifurcationConfig()


# ---------------------------------------------------------------------------
def _segment_sales(frame: pd.DataFrame) -> pd.Series:
    """How many stones this desk has actually SOLD in each segment.

    Sales, not listings: how often they turn a segment over is a different
    question from how long each one takes, and the desk's edge shows up exactly
    where the two disagree.
    """
    return frame[frame["event"] == 1].groupby("segment", observed=True).size()


def classify_stones(stock: pd.DataFrame, model, *, frame: pd.DataFrame | None = None,
                    depth_table=None, cfg: BifurcationConfig | None = None
                    ) -> pd.DataFrame:
    """One row per stock stone: its class, its ageing bucket, and its basis.

    `stock` is the Stock slice of the survival frame; `frame` is the whole frame
    (used only to count how many sales stand behind each segment). `depth_table`
    is optional — without it the market-depth columns are None and say so,
    rather than being quietly filled with zeros.
    """
    cfg = cfg or serving_bifurcation_config()
    frame = stock if frame is None else frame
    sales = _segment_sales(frame)

    pred = model.predict_remaining_days(stock)
    out = pd.DataFrame({
        "StoneId": pred["StoneId"],
        "Segment": pred["segment"],
        "AgeDays": pred["age_days"],
        "AgeingBucket": [ageing_bucket(a) for a in pred["age_days"]],
        "ExpectedDaysToSell": pred["expected_remaining_days"],
        "ExpectedDaysLow": pred["remaining_low"],
        "ExpectedDaysHigh": pred["remaining_high"],
        "ExpectedTotalDays": pred["expected_total_days"],
        "OwnVelocityScore": pred["own_velocity_score"],
        "HorizonLimited": pred["horizon_limited"],
    })
    out["Class"] = [cfg.label(s) for s in out["OwnVelocityScore"]]
    out["ClassFrontOffice"] = [to_frontoffice(c) for c in out["Class"]]
    out["SegmentSales"] = out["Segment"].map(sales).fillna(0).astype(int)
    out["ThinSegment"] = out["SegmentSales"] < cfg.min_segment_sales
    out["RedFlag"] = out["AgeingBucket"] == RED_FLAG_BUCKET

    # --- market depth: the SECOND number, never merged (MOU 5.2 / 8.1) --------
    from .market_depth import own_vs_market
    if depth_table is not None:
        depths = [depth_table.get(s) for s in out["Segment"]]
        out["MarketDepth"] = [d.depth for d in depths]
        out["MarketDepthScore"] = [d.score for d in depths]
        out["MarketDepthBasis"] = [d.basis for d in depths]
    else:
        out["MarketDepth"] = None
        out["MarketDepthScore"] = None
        out["MarketDepthBasis"] = "market depth not looked up for this run"
    pairs = [own_vs_market(v, d) for v, d in
             zip(out["OwnVelocityScore"], out["MarketDepthScore"])]
    out["VelocityRatio"] = [p["velocity_ratio"] for p in pairs]
    out["OwnVsMarket"] = [p["edge"] for p in pairs]
    out["OwnVsMarketBasis"] = [p["edge_basis"] for p in pairs]

    out["ClassBasis"] = [
        _basis(row) for row in out.to_dict("records")
    ]
    return out


def _basis(r: dict) -> str:
    """The sentence under the class. MOU 10.3: never a bare label."""
    parts = [f"velocity score {r['OwnVelocityScore']:.0f}/100 "
             f"(percentile of this desk's own days-to-sell)"]
    if pd.notna(r["ExpectedDaysToSell"]):
        parts.append(f"expected {r['ExpectedDaysToSell']:.0f}d more"
                     + (f" ({r['ExpectedDaysLow']:.0f}-{r['ExpectedDaysHigh']:.0f}d)"
                        if pd.notna(r["ExpectedDaysLow"]) and pd.notna(r["ExpectedDaysHigh"])
                        else ""))
    else:
        parts.append("median not reached inside the observation window")
    parts.append(f"{r['AgeDays']:.0f}d old ({r['AgeingBucket']})")
    parts.append(f"{r['SegmentSales']} own sales in {r['Segment']}"
                 + (" — THIN, judged on a coarser norm" if r["ThinSegment"] else ""))
    if r.get("HorizonLimited"):
        parts.append("estimate truncated by the observation window — read as a floor")
    return "; ".join(parts)


def classify_segments(frame: pd.DataFrame, model, *, depth_table=None,
                      cfg: BifurcationConfig | None = None) -> pd.DataFrame:
    """The same five classes at SEGMENT level (MOU 5.1 asks for both).

    A segment is scored from its stones' expected days at LISTING — not
    conditional on age — because the question here is "what does this kind of
    goods do?", not "what will this particular stone do next".
    """
    cfg = cfg or serving_bifurcation_config()
    pred = model.predict_days(frame)
    sales = _segment_sales(frame)

    joined = frame[["segment", "Status"]].reset_index(drop=True).assign(
        expected_days=pred["expected_days"].to_numpy(),
        score=pred["own_velocity_score"].to_numpy())
    g = joined.groupby("segment", observed=True)
    out = pd.DataFrame({
        "Segment": g.size().index,
        "Stones": g.size().to_numpy(),
        "InStock": g.apply(lambda x: int((x["Status"] == "Stock").sum()),
                           include_groups=False).to_numpy(),
        "ExpectedDaysToSell": g["expected_days"].median().round(1).to_numpy(),
        "OwnVelocityScore": g["score"].median().round(0).to_numpy(),
    })
    out["SegmentSales"] = out["Segment"].map(sales).fillna(0).astype(int)
    out["Class"] = [cfg.label(s) for s in out["OwnVelocityScore"]]
    out["ClassFrontOffice"] = [to_frontoffice(c) for c in out["Class"]]
    out["ThinSegment"] = out["SegmentSales"] < cfg.min_segment_sales

    from .market_depth import own_vs_market
    if depth_table is not None:
        depths = [depth_table.get(s) for s in out["Segment"]]
        out["MarketDepth"] = [d.depth for d in depths]
        out["MarketDepthScore"] = [d.score for d in depths]
    else:
        out["MarketDepth"] = None
        out["MarketDepthScore"] = None
    pairs = [own_vs_market(v, d) for v, d in
             zip(out["OwnVelocityScore"], out["MarketDepthScore"])]
    out["VelocityRatio"] = [p["velocity_ratio"] for p in pairs]
    out["OwnVsMarket"] = [p["edge"] for p in pairs]

    out["Basis"] = [
        (f"{n} stones ({s} own sales); median expected {d:.0f}d to sell"
         + ("; THIN — treat the direction as indicative" if t else ""))
        for n, s, d, t in zip(out["Stones"], out["SegmentSales"],
                              out["ExpectedDaysToSell"].fillna(-1), out["ThinSegment"])
    ]
    return out.sort_values("OwnVelocityScore", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# the external prior — labelled, never dressed up as measured
# ---------------------------------------------------------------------------
# Trade rule of thumb for jewellery inventory. It is NOT derived from this
# client's data and is never used to move a number; it is a smell test on our
# own output, presented the way `market/context.py` presents macro priors.
TRADE_BENCHMARK = {
    "turns_per_year_low": 0.7,
    "turns_per_year_high": 1.2,
    "max_share_older_than_120d": 0.20,
    "source": "general jewellery-trade rule of thumb (EXTERNAL PRIOR — not "
              "measured from Glow Star's data; see MOU 5.4 on labelled priors)",
}


def benchmark_check(classified: pd.DataFrame, *, value_col: str | None = None,
                    realized_median_days: float | None = None) -> dict:
    """Smell-test the classified book against the trade prior.

    Used as intended, this caught a real defect: it fired at 59% of stock in
    "Slow" and the cause was a scoring bug, not the desk.

    But a prior that fires is not automatically right, and there are two very
    different reasons it can. Pass `realized_median_days` — the Kaplan-Meier
    median of the client's OWN realized sales, which is data with no model in
    it — and the check can tell them apart:

      * model far from the prior AND far from the realized median  -> our bug;
      * model close to the realized median, both far from the prior -> the prior
        does not describe this business.

    The second is the live answer here. The 0.7-1.2 turns figure is a RETAIL
    jewellery statistic; Glow Star is a wholesale natural-diamond desk whose own
    records show a median days-to-sell around a month. Reporting "outside the
    benchmark" without that distinction would tell the client their own verified
    turnover is a defect.
    """
    n = len(classified)
    if not n:
        return {"verdict": "no stock to check", **TRADE_BENCHMARK}
    days = classified["ExpectedDaysToSell"].to_numpy(float)
    usable = days[np.isfinite(days) & (days > 0)]
    turns = float(365.0 / np.median(usable)) if len(usable) else None

    older = classified["AgeDays"].to_numpy(float) > 120
    if value_col and value_col in classified:
        v = classified[value_col].to_numpy(float)
        share = float(np.nansum(v[older]) / max(np.nansum(v), 1e-9))
        share_basis = f"by value ({value_col})"
    else:
        share = float(older.mean())
        share_basis = "by stone count (no value column supplied)"

    within_turns = (turns is not None
                    and TRADE_BENCHMARK["turns_per_year_low"] <= turns
                    <= TRADE_BENCHMARK["turns_per_year_high"])
    within_age = share <= TRADE_BENCHMARK["max_share_older_than_120d"]

    realized_turns = (None if not realized_median_days or realized_median_days <= 0
                      else round(365.0 / realized_median_days, 2))
    if within_turns and within_age:
        verdict = "consistent with the trade prior"
    elif realized_turns is None:
        verdict = ("outside the trade prior — check the model before the desk "
                   "(pass realized_median_days to tell a model error from a "
                   "prior that does not fit this business)")
    elif turns is not None and abs(turns - realized_turns) <= 0.35 * realized_turns:
        verdict = (f"outside the trade prior, but the model agrees with the "
                   f"client's OWN realized sales ({realized_turns} turns/yr from "
                   f"a {realized_median_days:.0f}-day Kaplan-Meier median). The "
                   f"prior is a RETAIL jewellery figure and does not describe "
                   f"this wholesale desk — do not report it as a defect.")
    else:
        verdict = (f"outside the trade prior AND away from the client's own "
                   f"realized sales ({realized_turns} turns/yr) — suspect the "
                   f"model.")

    return {
        "implied_turns_per_year": None if turns is None else round(turns, 2),
        "realized_turns_per_year": realized_turns,
        "share_older_than_120d": round(share, 3),
        "share_basis": share_basis,
        "within_turns_benchmark": within_turns,
        "within_ageing_benchmark": within_age,
        "verdict": verdict,
        **TRADE_BENCHMARK,
    }


def compare_with_live_field(classified: pd.DataFrame) -> dict:
    """Before/after on the LIVE FrontOffice Tradeability field (MOU 2.2).

    The desk's screen is bound to `service/tradeability.py` today. MOU 2.2 is
    explicit that improving this number changes something they are already
    reading, so any change ships with a before/after on the same stones and the
    desk is TOLD. This function is that before/after, as a command rather than a
    figure in a document, so it can be re-run the day they are briefed.

    Measured 2026-08-27 on the whole live stock book (10,683 stones):
    only 22.8% of labels would be unchanged, and the movement is almost entirely
    one way — 7,762 stones slower, 488 faster.

    That gap is not a defect in either estimate; the two answer different
    questions. The live field applies a segment's median time FROM LISTING to a
    stone that has ALREADY sat unsold, and stock is by construction the
    surviving tail — the fast stones in that segment have gone. The velocity
    model conditions on the days already elapsed, so it says a 21-day-old stone
    in a 30-day segment is slower than average, because it demonstrably is.

    The model's answer is the better one, and it is still not switched
    unilaterally: MOU 9.1 requires written notice in both directions on a field
    change, and moving a number the desk has learned to read without telling
    them is worse than leaving it alone.
    """
    from ..service import tradeability as T

    c = classified[classified.get("VelocityEstimated", True).astype(bool)
                   if "VelocityEstimated" in classified else slice(None)].copy()
    parts = c["Segment"].str.split("|", expand=True)
    live = [T.tradeability_for(s, 1.0, co, cl)
            for s, co, cl in zip(parts[0], parts[2], parts[3])]
    c["LiveLabel"] = [x["label"] for x in live]
    c["LiveDays"] = [x["median_days"] for x in live]
    c["NewLabel"] = c["ClassFrontOffice"]

    order = list(T.LABELS)
    rank = {l: i for i, l in enumerate(order)}
    moved = c[c["LiveLabel"] != c["NewLabel"]]
    jumps = [rank.get(b, 0) - rank.get(a, 0)
             for a, b in zip(moved["LiveLabel"], moved["NewLabel"])]
    return {
        "stones": int(len(c)),
        "unchanged_share": round(float((c["LiveLabel"] == c["NewLabel"]).mean()), 4),
        "moved": int(len(moved)),
        "moved_slower": int(sum(1 for j in jumps if j > 0)),
        "moved_faster": int(sum(1 for j in jumps if j < 0)),
        "median_days_live": (None if c["LiveDays"].isna().all()
                             else float(np.nanmedian(c["LiveDays"]))),
        "median_days_model": (None if c["ExpectedDaysToSell"].isna().all()
                              else float(c["ExpectedDaysToSell"].median())),
        "matrix": pd.crosstab(c["LiveLabel"], c["NewLabel"])
                    .reindex(index=order, columns=order).fillna(0).astype(int),
        "recommendation": (
            "Do NOT switch the live field silently. The model's estimate is the "
            "better one — it conditions on the days a stone has already sat "
            "unsold, which the segment median cannot — but this moves a number "
            "the desk reads every day. Agree the cutover with the client's IT/API "
            "owner (MOU 2, Jay Bhai) with this table in front of them, keep the "
            "vocabulary mapping in place until their screen is switched, and only "
            "then delete it (MOU 9.1)."),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import sys
    from .survival import build_survival_frame
    from .velocity import VelocityModel

    frame, rep = build_survival_frame()
    print(rep.summary())
    model = VelocityModel().fit(frame)
    stock = frame[frame["Status"] == "Stock"]
    stones = classify_stones(stock, model, frame=frame)

    print("\nSTOCK BY CLASS (MOU wording; FrontOffice words in brackets)")
    counts = stones["Class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    for c in CLASSES:
        print(f"  {c:<10} [{FRONTOFFICE_LABELS[c]:<10}] {counts[c]:>6}  "
              f"({counts[c] / len(stones):.1%})")
    print("\nAGEING BUCKETS")
    ab = stones["AgeingBucket"].value_counts().reindex([b[2] for b in AGEING_BUCKETS]).fillna(0).astype(int)
    for _, _, name in AGEING_BUCKETS:
        flag = "   <- red flag" if name == RED_FLAG_BUCKET and ab[name] else ""
        print(f"  {name:<10} {ab[name]:>6}  ({ab[name] / len(stones):.1%}){flag}")
    print("\nCROSS-TAB (class x ageing) — the pair is the point:")
    print(pd.crosstab(stones["Class"].astype(str), stones["AgeingBucket"].astype(str))
          .reindex(CLASSES).fillna(0).astype(int).to_string())
    print("\nBENCHMARK CHECK (external prior, advisory)")
    from .survival import km_median
    realized = km_median(frame["duration"], frame["event"].astype(bool))
    for k, v in benchmark_check(stones, realized_median_days=realized).items():
        print(f"  {k}: {v}")
    print("\nEXAMPLE BASIS:")
    print(" ", stones.iloc[0]["ClassBasis"])

    if "--compare-live" in sys.argv:
        cmp = compare_with_live_field(stones)
        print("\n" + "=" * 66)
        print("BEFORE / AFTER ON THE LIVE FRONTOFFICE TRADEABILITY FIELD (MOU 2.2)")
        print("=" * 66)
        print(f"  stones compared     : {cmp['stones']:,}")
        print(f"  label unchanged     : {cmp['unchanged_share']:.1%}")
        print(f"  moved slower/faster : {cmp['moved_slower']:,} / {cmp['moved_faster']:,}")
        print(f"  median days         : live {cmp['median_days_live']:.0f} "
              f"-> model {cmp['median_days_model']:.0f}")
        print("\n  rows = today's field, cols = velocity model")
        print(cmp["matrix"].to_string())
        print("\n  " + cmp["recommendation"])


if __name__ == "__main__":
    main()
