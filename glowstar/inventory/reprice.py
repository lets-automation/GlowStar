"""Velocity-adjusted repricing — Workstream B, Phase C (MOU 5.1, 5.3).

The pricing engine gives the FAIR price. This layer proposes a move on top of
it: shallower on fast movers to capture margin, deeper on stale goods to free
capital. It never prices a stone itself and never writes anywhere — it proposes,
and a human at the desk decides (MOU 11.3, 11.9).

BOUNDED BY THE SHIPPED BAND — NOT A CONVENIENT ONE
---------------------------------------------------
Every proposed discount is clamped inside the pricing engine's own calibrated
interval for that stone, and that interval must come from the pipeline the
client actually receives. `_assert_band_is_the_shipped_one()` fails loudly
otherwise. This is CLAUDE.md Trap 5, which has already cost this project once:
the conformal band was calibrated on a model that was not being served, so the
published "80% band" was a promise about a function nobody ever ran.

THE DAY EFFECT OF A PRICE MOVE IS NOT IDENTIFIABLE FROM THIS DATA
------------------------------------------------------------------
The brief asks for "expected change in days-to-sell ... from the survival
model's covariate sensitivity". That was built, measured, and does not work, and
shipping it would have put a number with the wrong sign in front of the desk.
Three measurements, in order:

1. The velocity model's sensitivity to its price covariate runs BACKWARDS. Shift
   `base_discount` 6 points deeper and predicted days-to-sell goes UP by 2.3.
   The covariate is not a lever, it is a LABEL: goods that need a deep base
   discount are the hard-to-move ones, so "deep" marks slow rather than causing
   fast.

2. Ablating it changes nothing out-of-time — C-index 0.5924 with, 0.5929
   without. It carries no independent velocity signal at all.

3. The real natural experiment, from 15 daily-snapshot intervals: 11,963 stones
   whose ASKING discount was actually cut while unsold, against 71,780 unchanged
   controls, stratified on age x size band x shape. P(sold within 30 days),
   control base 0.32:

       cut 1-3 pts   +0.026   95% CI [+0.009, +0.041]
       cut 3-5 pts   -0.044   95% CI [-0.074, -0.006]

   A bigger cut associates with a SLOWER sale, significantly. That is not a
   dose-response curve; it is the confounder showing through — the desk cuts
   hardest on what it already cannot move, and age/size/shape does not capture
   "hard to move". No causal elasticity is identified here, in either direction.

So `ProjectedDaysChange` is None on every suggestion, with the reason attached.
The margin consequence IS exact arithmetic and is reported to the dollar. What
would settle it is cheap and specific, and it is in the output: **a deliberate
price test** — the desk picks matched pairs of comparable stones of similar age
and cuts one of each, chosen by us rather than by how stuck the stone is. A few
hundred pairs over a month identifies the elasticity that this observational
data cannot.

ON GMROI (MOU 5.3)
------------------
GMROI is gross margin over average inventory cost, and the client's feed carries
no cost field — `BasePriceDiscount` is a list basis, sitting SHALLOWER than the
sale price, not a cost. Two consequences, both stated rather than papered over:

  * true GMROI is computed only when a cost basis is supplied (`cost_discount`);
  * without one, the guard is the DISCIPLINE rather than the ratio: a move may
    give up only a bounded amount of margin, deeper moves are reserved for goods
    that are genuinely stale, and no move is proposed on the strength of a
    turnover gain that has not been demonstrated. "Sell everything fast at a
    deep discount" would raise turnover and destroy margin, which MOU 5.3 rules
    out explicitly.

Run:  python -m glowstar.inventory.reprice
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import SETTINGS
from .bifurcate import CLASSES

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepriceConfig:
    """Every knob that moves a proposed price. Same rule as `serving_config()`.

    Moves are in DISCOUNT POINTS and signed the way the data is: the discount is
    negative, so a POSITIVE move is shallower (dearer, more margin) and a
    NEGATIVE move is deeper (cheaper, freeing capital).
    """

    move_by_class: dict[str, float] = field(default_factory=lambda: {
        "Fast": +1.5,        # it sells anyway — take the margin
        "Semi-Fast": +0.75,
        "Medium": 0.0,       # the fair price is the answer
        "Semi-Slow": -0.75,
        "Slow": -1.5,
    })
    # Staleness GATES a deeper move, it does not merely scale one — note the
    # 0.0. A stone in the slowest quintile that has been on the shelf for three
    # weeks has not demonstrated a problem; discounting it buys nothing and
    # costs certain margin.
    #
    # This started as a plain multiplier (1.0 / 1.5 / 2.0 / 2.5) and the first
    # run priced the consequence: across 1,500 live stock stones it gave up
    # $118,180 of margin to capture $2,480 — 47 to 1 — in exchange for a
    # turnover gain that the elasticity work above shows is not demonstrable.
    # That is the failure MOU 5.3 names in as many words. With deeper moves
    # gated on real staleness the same book gives up a small fraction of it, and
    # every dollar of it is spent on goods that have actually stalled.
    #
    # Shallower moves are NOT gated: taking margin on something the model says
    # is moving is the side of the trade with evidence behind it.
    ageing_multiplier: dict[str, float] = field(default_factory=lambda: {
        "0-90": 0.0, "91-180": 1.0, "181-365": 1.5, "365+": 2.0,
    })
    # Hard ceiling on how much margin one move may give up, whatever the class
    # and ageing multiplier work out to. Without a demonstrated turnover gain,
    # an unbounded "just discount it" is exactly the failure MOU 5.3 names.
    max_deeper_pts: float = 4.0
    max_shallower_pts: float = 2.5
    # A band wider than this means the engine is not confident about this stone;
    # it goes to a human rather than being moved automatically.
    low_confidence_band_pts: float = 12.0
    high_value_usd: float = SETTINGS.high_value_usd
    # The desk's cost as a discount off Rap, if they ever supply one. Until then
    # GMROI is reported as not computable rather than guessed.
    cost_discount: float | None = None
    # Stones older than this are eligible for the liquidation flag.
    liquidation_age_days: float = 180.0
    liquidation_classes: tuple[str, ...] = ("Slow", "Semi-Slow")


def serving_reprice_config() -> RepriceConfig:
    """THE repricing config. A knob outside it cannot reach a proposal."""
    return RepriceConfig()


def _assert_band_is_the_shipped_one(engine) -> None:
    """The band bounding every move must come from the pipeline that ships.

    An assertion, not a log line: a silent divergence here is what let an
    unmeasured pricing path reach the desk for weeks.
    """
    from ..training.retrain import serving_config

    ref = serving_config(getattr(engine.cfg, "split_date", None))
    for f in ("market_led", "anchor_lambda", "apply_asking_offset", "use_trend"):
        got, want = getattr(engine.cfg, f), getattr(ref, f)
        if got != want:
            raise AssertionError(
                f"Repricing is bounding moves with {f}={got!r} but serving uses "
                f"{want!r}. The band must come from the config that ships "
                f"(see training.retrain.serving_config).")


# ---------------------------------------------------------------------------
def _review_reasons(r: dict, cfg: RepriceConfig) -> list[str]:
    """Why a human must look at this one. MOU 11.9: no auto-apply on these."""
    out = []
    if r["CurrentNet"] is not None and r["CurrentNet"] >= cfg.high_value_usd:
        out.append(f"high value (>= ${cfg.high_value_usd:,.0f})")
    if r["BandWidthPts"] is not None and r["BandWidthPts"] > cfg.low_confidence_band_pts:
        out.append(f"low confidence (band {r['BandWidthPts']:.1f} pts wide)")
    if r.get("ThinSegment"):
        out.append("thin segment — velocity judged on a coarser norm")
    if r.get("HorizonLimited"):
        out.append("days-to-sell truncated by the observation window")
    for flag in (r.get("Flags") or []):
        if flag in ("fluor_review", "bgm_review", "rare_shape", "fancy_color",
                    "thin_market", "no_grid_cell"):
            out.append(f"engine flag: {flag}")
    return out


def propose(classified: pd.DataFrame, priced: pd.DataFrame, *,
            cfg: RepriceConfig | None = None) -> pd.DataFrame:
    """One repricing proposal per stone, with its basis and its review flags.

    `classified` is `bifurcate.classify_stones()` output. `priced` carries the
    engine's own answer per stone: StoneId, FairDiscount, CiLow, CiHigh, Rap,
    Weight, and optionally Flags. Both are joined on StoneId; a stone missing
    from either side is dropped and counted, never silently defaulted.
    """
    cfg = cfg or serving_reprice_config()
    df = classified.merge(priced, on="StoneId", how="inner", suffixes=("", "_p"))
    if len(df) < len(classified):
        log.info("Repricing: %d of %d classified stones had no engine price and "
                 "were left out.", len(classified) - len(df), len(classified))

    fair = pd.to_numeric(df["FairDiscount"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["CiLow"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(df["CiHigh"], errors="coerce").to_numpy(float)
    rap = pd.to_numeric(df["Rap"], errors="coerce").to_numpy(float)
    wt = pd.to_numeric(df["Weight"], errors="coerce").to_numpy(float)

    base_move = np.array([cfg.move_by_class.get(c, 0.0) for c in df["Class"]])
    mult = np.array([cfg.ageing_multiplier.get(b, 1.0) for b in df["AgeingBucket"]])
    # Staleness only ever deepens. See the config comment.
    move = np.where(base_move < 0, base_move * mult, base_move)
    move = np.clip(move, -cfg.max_deeper_pts, cfg.max_shallower_pts)

    proposed_raw = fair + move
    # BOUNDED BY THE SHIPPED BAND. `lo` is the deepest the engine will defend,
    # `hi` the shallowest.
    proposed = np.clip(proposed_raw, np.minimum(lo, hi), np.maximum(lo, hi))
    clamped = np.abs(proposed - proposed_raw) > 1e-9

    cur_ppc = rap * (1.0 + fair / 100.0)
    new_ppc = rap * (1.0 + proposed / 100.0)
    cur_net = cur_ppc * wt
    new_net = new_ppc * wt

    out = pd.DataFrame({
        "StoneId": df["StoneId"],
        "Segment": df["Segment"],
        "Class": df["Class"],
        "ClassFrontOffice": df["ClassFrontOffice"],
        "AgeDays": df["AgeDays"],
        "AgeingBucket": df["AgeingBucket"],
        "ExpectedDaysToSell": df["ExpectedDaysToSell"],
        "OwnVelocityScore": df["OwnVelocityScore"],
        "MarketDepthScore": df.get("MarketDepthScore"),
        "VelocityRatio": df.get("VelocityRatio"),
        "OwnVsMarket": df.get("OwnVsMarket"),
        "FairDiscount": np.round(fair, 2),
        "ProposedDiscount": np.round(proposed, 2),
        "MovePts": np.round(proposed - fair, 2),
        "Direction": np.where(proposed > fair, "shallower (capture margin)",
                              np.where(proposed < fair, "deeper (free capital)", "no change")),
        "BandLow": np.round(lo, 2), "BandHigh": np.round(hi, 2),
        "BandWidthPts": np.round(np.abs(hi - lo), 2),
        "ClampedToBand": clamped,
        "CurrentPpc": np.round(cur_ppc, 2), "ProposedPpc": np.round(new_ppc, 2),
        "CurrentNet": np.round(cur_net, 2), "ProposedNet": np.round(new_net, 2),
        "RevenueChangeUsd": np.round(new_net - cur_net, 2),
        # NOT PREDICTED — see the module docstring. Emitting a number here was
        # measured to require an elasticity this data cannot identify.
        "ProjectedDaysChange": None,
        "ProjectedDaysBasis": (
            "not predicted: no causal price-to-speed elasticity is identifiable "
            "from this data. Observationally a 1-3pt cut is worth +2.6pts of "
            "30-day sale probability but a 3-5pt cut is worth -4.4pts, which is "
            "the confounder (the desk cuts hardest on the hardest goods), not a "
            "dose response. A deliberate matched-pair price test would settle it."),
        "ThinSegment": df.get("ThinSegment", False),
        "HorizonLimited": df.get("HorizonLimited", False),
        "Flags": df.get("Flags"),
    })

    # --- GMROI, only where a cost basis makes it real ------------------------
    if cfg.cost_discount is not None:
        cost = rap * (1.0 + cfg.cost_discount / 100.0) * wt
        with np.errstate(divide="ignore", invalid="ignore"):
            out["GmroiCurrent"] = np.round((cur_net - cost) / np.where(cost > 0, cost, np.nan), 3)
            out["GmroiProposed"] = np.round((new_net - cost) / np.where(cost > 0, cost, np.nan), 3)
        out["GmroiBasis"] = (f"cost taken as {cfg.cost_discount:.1f}% off Rap, as "
                             f"supplied by the caller — not a measured cost")
    else:
        out["GmroiCurrent"] = None
        out["GmroiProposed"] = None
        out["GmroiBasis"] = ("not computable — the client's feed carries no cost "
                             "field (BasePriceDiscount is a list basis, shallower "
                             "than the sale price). Supply cost_discount to enable it.")

    reasons = [_review_reasons(r, cfg) for r in out.to_dict("records")]
    out["ReviewReasons"] = ["; ".join(x) for x in reasons]
    out["NeedsHumanReview"] = [bool(x) for x in reasons]
    # MOU 11.9: nothing here is ever applied automatically. The column exists so
    # a caller cannot mistake "no review reason" for "safe to push".
    out["AutoApply"] = False
    out["LiquidationCandidate"] = (
        (out["AgeDays"] >= cfg.liquidation_age_days)
        & out["Class"].isin(cfg.liquidation_classes))

    out["Why"] = [_why(r, cfg) for r in out.to_dict("records")]
    return out


def _why(r: dict, cfg: RepriceConfig) -> str:
    """The sentence the desk reads. Templated: every number is already computed."""
    if abs(r["MovePts"]) < 0.01:
        return (f"{r['Class']} at {r['AgeDays']:.0f} days — the fair price is the "
                f"answer; no velocity adjustment proposed.")
    verb = "hold back" if r["MovePts"] > 0 else "give"
    tail = (f" Clamped to the engine's {r['BandLow']:.1f}..{r['BandHigh']:.1f} "
            f"confidence band." if r["ClampedToBand"] else "")
    liq = (" Old enough and slow enough to consider liquidating rather than "
           "repricing." if r["LiquidationCandidate"] else "")
    return (f"{r['Class']}, {r['AgeDays']:.0f} days old ({r['AgeingBucket']}) — "
            f"{verb} {abs(r['MovePts']):.1f} pts vs the fair price "
            f"({r['FairDiscount']:.1f}% -> {r['ProposedDiscount']:.1f}%), "
            f"{'+' if r['RevenueChangeUsd'] >= 0 else '-'}"
            f"${abs(r['RevenueChangeUsd']):,.0f} on this stone.{tail}{liq}")


def summarise(proposals: pd.DataFrame) -> dict:
    """Book-level effect of the proposals. Deterministic arithmetic only."""
    if not len(proposals):
        return {"stones": 0}
    rev = proposals["RevenueChangeUsd"].to_numpy(float)
    return {
        "stones": int(len(proposals)),
        "moved": int((proposals["MovePts"].abs() >= 0.01).sum()),
        "shallower": int((proposals["MovePts"] > 0).sum()),
        "deeper": int((proposals["MovePts"] < 0).sum()),
        "clamped_to_band": int(proposals["ClampedToBand"].sum()),
        "needs_human_review": int(proposals["NeedsHumanReview"].sum()),
        "liquidation_candidates": int(proposals["LiquidationCandidate"].sum()),
        "revenue_change_usd": round(float(np.nansum(rev)), 2),
        "margin_given_up_usd": round(float(-np.nansum(rev[rev < 0])), 2),
        "margin_captured_usd": round(float(np.nansum(rev[rev > 0])), 2),
        "turnover_effect": ("not quantified — no causal price-to-speed elasticity "
                            "is identifiable from observational data; see "
                            "ProjectedDaysBasis"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from ..models import registry
    from .bifurcate import classify_stones
    from .survival import build_survival_frame
    from .velocity import VelocityModel

    frame, rep = build_survival_frame()
    print(rep.summary())
    model = VelocityModel().fit(frame)
    stock = frame[frame["Status"] == "Stock"]
    classified = classify_stones(stock, model, frame=frame)

    engine, card = registry.load_current()
    if engine is None:
        print("No promoted model in the registry — run the retrain first.")
        return
    _assert_band_is_the_shipped_one(engine)
    print(f"Bounding moves with the SHIPPED band from model {(card or {}).get('version')}")

    from ..data.loaders import load_records, stock_stones
    df, _ = load_records()
    rows = stock_stones(df)
    # A RANDOM sample, not head(): records.json comes back oldest-stock-first, so
    # the first 1,500 rows are heavily stale and every aggregate read from them
    # overstates how much of the book needs discounting.
    rows = rows[rows["StoneId"].astype(str).isin(set(classified["StoneId"]))]
    rows = rows.sample(min(1500, len(rows)), random_state=42)
    sugg = engine.predict(rows)
    priced = pd.DataFrame({
        "StoneId": rows["StoneId"].astype(str).to_numpy(),
        "FairDiscount": [s.suggested_discount for s in sugg],
        "CiLow": [s.ci_discount_low for s in sugg],
        "CiHigh": [s.ci_discount_high for s in sugg],
        "Rap": pd.to_numeric(rows["Rap"], errors="coerce").to_numpy(),
        "Weight": pd.to_numeric(rows["Weight"], errors="coerce").to_numpy(),
        "Flags": [s.flags for s in sugg],
    })
    out = propose(classified, priced)
    print(f"\nproposals: {len(out)} (random sample of the stock book)")
    print("  ageing mix of the sample: "
          + ", ".join(f"{k} {v:.0%}" for k, v in
                      out["AgeingBucket"].value_counts(normalize=True).items()))
    for k, v in summarise(out).items():
        print(f"  {k}: {v}")
    print("\nBY CLASS:")
    print(out.groupby("Class")[["MovePts", "RevenueChangeUsd"]]
          .agg(["size", "mean"]).round(2).to_string())
    print("\nEXAMPLES:")
    for _, r in out.sort_values("MovePts").head(2).iterrows():
        print(" -", r["Why"])
    for _, r in out.sort_values("MovePts", ascending=False).head(2).iterrows():
        print(" -", r["Why"])


if __name__ == "__main__":
    main()
