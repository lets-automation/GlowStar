"""Phase-C gate: five-class bifurcation, ageing buckets, and repricing.

The invariants that matter to the client, and the two that already bit:

  * the MOU vocabulary is canonical, and the LIVE FrontOffice words are still
    produced at the boundary — their screen is bound to them today (MOU 2.2/9.1);
  * a repricing move never leaves the pricing engine's SHIPPED confidence band,
    and never auto-applies on a high-value or low-confidence stone (MOU 11.9);
  * deeper discounts are GATED on real staleness. Ungated, the first build gave
    up $118,180 of margin to capture $2,480 across 1,500 live stones — the exact
    "turnover at any cost" failure MOU 5.3 rules out;
  * no suggestion carries a projected days-to-sell change, because no causal
    price-to-speed elasticity is identifiable from this data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glowstar.inventory import bifurcate as BF
from glowstar.inventory import reprice as RP


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
def test_mou_wording_is_canonical():
    assert BF.CLASSES == ("Fast", "Semi-Fast", "Medium", "Semi-Slow", "Slow")


def test_frontoffice_vocabulary_is_unchanged():
    """Their screen reads these words TODAY. Changing them is a coordinated
    change with the client's IT owner (MOU 9.1), not a refactor."""
    from glowstar.service.tradeability import LABELS

    assert LABELS == ("High", "Semi High", "Medium", "Semi Slow", "Slow")
    assert [BF.to_frontoffice(c) for c in BF.CLASSES] == list(LABELS)


def test_the_mapping_round_trips_both_ways():
    """History in the `scores` table carries the OLD words; a report that reads
    it must be able to show one vocabulary, not two."""
    for c in BF.CLASSES:
        assert BF.from_frontoffice(BF.to_frontoffice(c)) == c
    assert BF.from_frontoffice("High") == "Fast"
    assert BF.from_frontoffice("Semi High") == "Semi-Fast"
    assert BF.to_frontoffice(None) is None and BF.from_frontoffice(None) is None


def test_unknown_label_passes_through_rather_than_vanishing():
    assert BF.to_frontoffice("Unheard-Of") == "Unheard-Of"


# ---------------------------------------------------------------------------
# ageing buckets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("age,expected", [
    (0, "0-90"), (90, "0-90"), (91, "91-180"), (180, "91-180"),
    (181, "181-365"), (365, "181-365"), (366, "365+"), (5000, "365+"),
])
def test_ageing_buckets_are_the_mou_boundaries(age, expected):
    assert BF.ageing_bucket(age) == expected


def test_ageing_bucket_of_unknown_age_is_none_not_a_default():
    assert BF.ageing_bucket(None) is None
    assert BF.ageing_bucket(float("nan")) is None


def test_over_365_days_is_the_red_flag():
    assert BF.RED_FLAG_BUCKET == "365+"


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def test_five_classes_partition_the_score_range():
    cfg = BF.serving_bifurcation_config()
    labels = [cfg.label(s) for s in range(0, 101)]
    assert set(labels) == set(BF.CLASSES)
    assert cfg.label(100) == "Fast" and cfg.label(0) == "Slow"
    # monotone: a faster score is never a slower class
    order = {c: i for i, c in enumerate(BF.CLASSES)}
    idx = [order[x] for x in labels]
    assert all(a >= b for a, b in zip(idx, idx[1:]))


def test_missing_score_is_unclassified_rather_than_slow():
    cfg = BF.serving_bifurcation_config()
    assert cfg.label(None) is None
    assert cfg.label(float("nan")) is None


def test_benchmark_prior_is_labelled_as_a_prior():
    """MOU 5.4: priors are labelled, never dressed up as measured."""
    assert "EXTERNAL PRIOR" in BF.TRADE_BENCHMARK["source"]


def test_benchmark_distinguishes_a_model_error_from_a_prior_that_does_not_fit():
    """The check fired for real once (59% of stock in 'Slow' — a scoring bug).

    But this desk genuinely turns ~10x a year, so the retail 0.7-1.2 prior fires
    for a second, opposite reason. Reporting the client's own verified turnover
    as a defect would be worse than not checking at all.
    """
    fast_book = pd.DataFrame({"ExpectedDaysToSell": [35.0] * 50,
                              "AgeDays": [20.0] * 50})
    agrees = BF.benchmark_check(fast_book, realized_median_days=34.0)
    assert not agrees["within_turns_benchmark"]
    assert "do not report it as a defect" in agrees["verdict"]

    disagrees = BF.benchmark_check(fast_book, realized_median_days=300.0)
    assert "suspect the model" in disagrees["verdict"]

    unknown = BF.benchmark_check(fast_book)
    assert "pass realized_median_days" in unknown["verdict"]


# ---------------------------------------------------------------------------
# repricing
# ---------------------------------------------------------------------------
def _proposal_inputs(n=6, cls="Slow", age=200.0, fair=-50.0, lo=-54.0, hi=-46.0,
                     rap=5000.0, weight=1.0):
    classified = pd.DataFrame({
        "StoneId": [f"S{i}" for i in range(n)],
        "Segment": ["Round|5|G|VS1"] * n,
        "Class": [cls] * n,
        "ClassFrontOffice": [BF.to_frontoffice(cls)] * n,
        "AgeDays": [age] * n,
        "AgeingBucket": [BF.ageing_bucket(age)] * n,
        "ExpectedDaysToSell": [80.0] * n,
        "OwnVelocityScore": [10.0] * n,
        "MarketDepthScore": [None] * n,
        "VelocityRatio": [None] * n,
        "OwnVsMarket": [None] * n,
        "ThinSegment": [False] * n,
        "HorizonLimited": [False] * n,
    })
    priced = pd.DataFrame({
        "StoneId": [f"S{i}" for i in range(n)],
        "FairDiscount": [fair] * n, "CiLow": [lo] * n, "CiHigh": [hi] * n,
        "Rap": [rap] * n, "Weight": [weight] * n, "Flags": [[]] * n,
    })
    return classified, priced


def test_proposal_never_leaves_the_engines_band():
    """The Trap-5 rule as an assertion on output: a move is bounded by the band
    the client's own pipeline produced, however aggressive the policy is."""
    c, p = _proposal_inputs(cls="Slow", age=400.0, lo=-51.0, hi=-49.0)
    out = RP.propose(c, p)
    assert (out["ProposedDiscount"] >= -51.0 - 1e-9).all()
    assert (out["ProposedDiscount"] <= -49.0 + 1e-9).all()
    assert out["ClampedToBand"].all()      # the policy wanted more than the band allows


def test_a_narrow_band_wins_over_the_policy():
    c, p = _proposal_inputs(cls="Slow", age=400.0, fair=-50.0, lo=-50.0, hi=-50.0)
    out = RP.propose(c, p)
    assert (out["ProposedDiscount"] == -50.0).all()
    assert (out["MovePts"] == 0.0).all()


def test_fast_movers_get_shallower_and_slow_stale_get_deeper():
    fast, p = _proposal_inputs(cls="Fast", age=10.0)
    assert (RP.propose(fast, p)["MovePts"] > 0).all()
    slow, p2 = _proposal_inputs(cls="Slow", age=300.0)
    assert (RP.propose(slow, p2)["MovePts"] < 0).all()


def test_deeper_moves_are_gated_on_real_staleness_not_just_class():
    """The $118k lesson: a Slow stone three weeks old has not demonstrated a
    problem, and discounting it buys nothing while costing certain margin."""
    fresh, p = _proposal_inputs(cls="Slow", age=20.0)
    assert (RP.propose(fresh, p)["MovePts"] == 0.0).all()
    stale, p2 = _proposal_inputs(cls="Slow", age=200.0)
    assert (RP.propose(stale, p2)["MovePts"] < 0).all()


def test_staleness_never_makes_a_fast_stone_dearer():
    young, p = _proposal_inputs(cls="Fast", age=10.0)
    old, p2 = _proposal_inputs(cls="Fast", age=400.0)
    assert RP.propose(young, p)["MovePts"].iloc[0] == RP.propose(old, p2)["MovePts"].iloc[0]


def test_moves_are_capped_however_stale_the_stone():
    cfg = RP.serving_reprice_config()
    c, p = _proposal_inputs(cls="Slow", age=5000.0, lo=-90.0, hi=-10.0)
    out = RP.propose(c, p)
    assert (out["MovePts"] >= -cfg.max_deeper_pts - 1e-9).all()


def test_no_projected_days_change_is_ever_emitted():
    """No causal price-to-speed elasticity is identifiable from this data:
    observationally a 1-3pt cut is +2.6pts of 30-day sale probability while a
    3-5pt cut is -4.4pts. Emitting a number here would ship the confounder."""
    c, p = _proposal_inputs()
    out = RP.propose(c, p)
    assert out["ProjectedDaysChange"].isna().all()
    assert "not predicted" in out["ProjectedDaysBasis"].iloc[0]
    assert "price test" in out["ProjectedDaysBasis"].iloc[0]


def test_revenue_change_is_exact_arithmetic():
    c, p = _proposal_inputs(cls="Fast", age=10.0, fair=-50.0, lo=-54.0, hi=-40.0,
                            rap=5000.0, weight=2.0)
    out = RP.propose(c, p)
    move = out["MovePts"].iloc[0]
    assert out["RevenueChangeUsd"].iloc[0] == pytest.approx(5000.0 * 2.0 * move / 100.0, abs=0.01)


def test_high_value_and_low_confidence_always_go_to_a_human():
    """MOU 11.9: no price reaches a customer without a human approving it."""
    cfg = RP.serving_reprice_config()
    c, p = _proposal_inputs(cls="Fast", age=10.0, rap=cfg.high_value_usd, weight=10.0)
    out = RP.propose(c, p)
    assert out["NeedsHumanReview"].all()
    assert "high value" in out["ReviewReasons"].iloc[0]

    c2, p2 = _proposal_inputs(cls="Medium", age=10.0, lo=-70.0, hi=-30.0)
    out2 = RP.propose(c2, p2)
    assert out2["NeedsHumanReview"].all()
    assert "low confidence" in out2["ReviewReasons"].iloc[0]


def test_nothing_is_ever_auto_applied():
    """MOU 11.3: read-and-recommend only. The column exists so a caller cannot
    read 'no review reason' as 'safe to push'."""
    c, p = _proposal_inputs(cls="Medium", age=10.0)
    assert (RP.propose(c, p)["AutoApply"] == False).all()  # noqa: E712


def test_thin_segment_and_horizon_limit_are_surfaced_as_review_reasons():
    c, p = _proposal_inputs(cls="Slow", age=200.0)
    c["ThinSegment"] = True
    c["HorizonLimited"] = True
    out = RP.propose(c, p)
    assert "thin segment" in out["ReviewReasons"].iloc[0]
    assert "truncated" in out["ReviewReasons"].iloc[0]


def test_gmroi_is_not_computable_without_a_cost_basis():
    """The feed carries no cost field, and BasePriceDiscount is not one."""
    c, p = _proposal_inputs()
    out = RP.propose(c, p)
    assert out["GmroiCurrent"].isna().all()
    assert "no cost field" in out["GmroiBasis"].iloc[0]

    cfg = RP.RepriceConfig(cost_discount=-70.0)
    out2 = RP.propose(c, p, cfg=cfg)
    assert out2["GmroiCurrent"].notna().all()
    assert "not a measured cost" in out2["GmroiBasis"].iloc[0]


def test_summary_refuses_to_claim_a_turnover_lift():
    c, p = _proposal_inputs(cls="Slow", age=200.0)
    s = RP.summarise(RP.propose(c, p))
    assert "not quantified" in s["turnover_effect"]
    assert s["margin_given_up_usd"] > 0


def test_every_proposal_carries_a_readable_why():
    c, p = _proposal_inputs(cls="Slow", age=200.0)
    out = RP.propose(c, p)
    why = out["Why"].iloc[0]
    assert "Slow" in why and "200 days old" in why and "$" in why


def test_stones_missing_a_price_are_dropped_not_defaulted():
    c, p = _proposal_inputs(n=6)
    out = RP.propose(c, p.head(3))
    assert len(out) == 3


def test_band_assertion_rejects_a_non_serving_config():
    """Mirrors `_assert_gate_scores_what_ships`: bounding moves with a band from
    a configuration the client never receives is the incident, not the fix."""
    from glowstar.models.engine import EngineConfig

    class _Fake:
        cfg = EngineConfig(market_led=True)

    with pytest.raises(AssertionError, match="config that ships"):
        RP._assert_band_is_the_shipped_one(_Fake())

    class _Ok:
        from glowstar.training.retrain import serving_config as _sc
        cfg = _sc()

    RP._assert_band_is_the_shipped_one(_Ok())     # must not raise
