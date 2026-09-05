"""Phase-B gate: the velocity model, the C-index, and the own-vs-market pair.

The three things that must hold, and each has already been wrong once here:

  * the person-period expansion accounts for EXPOSURE (dropping a censored
    stone's partial final period biased every hazard upward);
  * survival is read at the right day (an off-by-one period reported P(sold by
    30d) as if it were day 45, and the C-index could not see it);
  * own velocity and market depth stay two numbers and a ratio (MOU 5.2/8.1).

The C-index is hand-checked against worked examples, because a concordance
implementation that is subtly wrong about ties still returns a plausible number
and nothing downstream would ever notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glowstar.inventory import market_depth as MD
from glowstar.inventory import survival as S
from glowstar.inventory import velocity as V
from glowstar.validation import survival_backtest as B


def _toy(n=600, seed=0) -> pd.DataFrame:
    """A small synthetic frame with a real signal, for the mechanical tests."""
    rng = np.random.default_rng(seed)
    shape = rng.choice(["Round", "Pear"], n)
    weight = rng.uniform(0.3, 2.0, n)
    # Rounds genuinely sell faster here, so a model that learns nothing fails.
    scale = np.where(shape == "Round", 20.0, 60.0)
    true_t = rng.exponential(scale)
    cens = rng.uniform(5, 120, n)
    dur = np.minimum(true_t, cens).round()
    ev = (true_t <= cens).astype(int)
    return pd.DataFrame({
        "StoneId": [f"T{i}" for i in range(n)],
        "Status": np.where(ev == 1, "Sold", "Stock"),
        "duration": dur, "event": ev,
        "entered": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 120, n), "D"),
        "shape": shape, "color": rng.choice(["D", "G", "J"], n),
        "clarity": rng.choice(["VS1", "SI1"], n), "cut_tier": "EX",
        "fluorescence": "NON", "lab": "GIA", "location": "IND",
        "weight": weight, "size_band": rng.integers(0, 10, n),
        "rap_ppc": rng.uniform(2000, 9000, n), "base_discount": rng.uniform(-60, -20, n),
        "brown_ord": -1.0, "milky_ord": 0.0, "shade_ord": 0.0, "green_ord": 0.0,
        "grid_discount": rng.uniform(-60, -20, n), "grid_age_days": rng.integers(0, 40, n),
        "listed_month": 1.0,
        "segment": shape + "|1|D|VS1",
    })


# ---------------------------------------------------------------------------
# person-period expansion
# ---------------------------------------------------------------------------
def test_expansion_keeps_a_same_day_sale():
    """`duration == 0` is common on this book and must not vanish."""
    f = pd.DataFrame({"duration": [0.0], "event": [1]})
    long = V.expand_periods(f, (0, 3, 7))
    assert len(long) == 1
    assert long["sold_in_period"].iloc[0] == 1
    assert long["period"].iloc[0] == 0


def test_event_lands_in_exactly_one_period():
    f = pd.DataFrame({"duration": [10.0], "event": [1]})
    long = V.expand_periods(f, (0, 3, 7, 14, 21))
    assert list(long["period"]) == [0, 1, 2]          # at risk in [0,3),[3,7),[7,14)
    assert list(long["sold_in_period"]) == [0, 0, 1]  # sells in [7,14)
    assert (long["exposure"] == 1.0).all()


def test_censored_partial_period_is_weighted_not_dropped():
    """The bug that inflated every hazard: exposure, not exclusion.

    A stone censored at day 10 was watched for 3 of the 7 days of the [7,14)
    period. Dropping that row removes a NON-EVENT from the denominator and
    biases the hazard up; carrying it at weight 3/7 is the actuarial correction.
    """
    f = pd.DataFrame({"duration": [10.0], "event": [0]})
    long = V.expand_periods(f, (0, 3, 7, 14, 21))
    assert list(long["period"]) == [0, 1, 2]
    assert (long["sold_in_period"] == 0).all()
    assert long["exposure"].iloc[0] == 1.0
    assert long["exposure"].iloc[1] == 1.0
    assert long["exposure"].iloc[2] == pytest.approx(3 / 7)


def test_censored_stone_contributes_no_events_at_all():
    f = pd.DataFrame({"duration": [100.0], "event": [0]})
    long = V.expand_periods(f, V.PERIOD_EDGES)
    assert long["sold_in_period"].sum() == 0
    assert len(long) > 0                       # it still informs the hazards


def test_exposure_weighting_moves_the_hazard_the_right_way():
    """Same data, with and without the weight: unweighted must be FASTER."""
    rng = np.random.default_rng(3)
    n = 400
    f = pd.DataFrame({"duration": rng.uniform(1, 100, n).round(),
                      "event": (rng.random(n) < 0.5).astype(int)})
    long = V.expand_periods(f, V.PERIOD_EDGES)
    events = long["sold_in_period"].sum()
    # What the model does now: every row counted for the part of the period we
    # actually watched.
    weighted = events / long["exposure"].sum()
    # What the first build did: partial rows dropped altogether, so only
    # full-exposure rows reached the denominator.
    dropped_partials = events / (long["exposure"] == 1.0).sum()
    assert dropped_partials > weighted, "the exposure weight is not doing anything"


# ---------------------------------------------------------------------------
# survival is read at the right day
# ---------------------------------------------------------------------------
def test_survival_is_read_at_the_day_not_the_end_of_its_period():
    """The off-by-one that predicted 0.99 against an observed 0.58.

    `surv[:, k]` is survival at `period_edges[k+1]`. Day 30 is the END of period
    4 with these edges, so it must read column 4 — reading the period that
    CONTAINS day 30 gives survival at day 45.
    """
    m = V.VelocityModel(V.VelocityConfig(period_edges=(0, 10, 20, 30, 40)))
    surv = np.array([[0.9, 0.8, 0.7, 0.6]])
    assert m._survival_at(surv, 0.0)[0] == 1.0     # nothing has elapsed
    assert m._survival_at(surv, 10.0)[0] == 0.9
    assert m._survival_at(surv, 30.0)[0] == 0.7
    assert m._survival_at(surv, 35.0)[0] == 0.7    # still day 30's boundary
    assert m._survival_at(surv, 40.0)[0] == 0.6


def test_crossing_interpolates_inside_the_period():
    m = V.VelocityModel(V.VelocityConfig(period_edges=(0, 10, 20)))
    surv = np.array([[0.8, 0.4]])                  # 0.5 falls inside [10, 20)
    d = m._crossing(surv, 0.5)[0]
    assert 10.0 < d < 20.0
    assert d == pytest.approx(10 + (0.8 - 0.5) / (0.8 - 0.4) * 10)


def test_median_not_reached_is_reported_not_invented():
    m = V.VelocityModel(V.VelocityConfig(period_edges=(0, 10, 20)))
    surv = np.array([[0.95, 0.9]])                 # never falls to 0.5
    assert not np.isfinite(m._crossing(surv, 0.5)[0])


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def test_model_learns_a_real_signal():
    f = _toy()
    m = V.VelocityModel().fit(f)
    days = m.predict_days(f)
    rounds = days.loc[f["shape"].to_numpy() == "Round", "expected_days"].mean()
    pears = days.loc[f["shape"].to_numpy() == "Pear", "expected_days"].mean()
    assert rounds < pears, "the model did not learn the shape effect it was given"


def test_all_missing_covariate_degrades_instead_of_crashing():
    """CLAUDE.md Trap 4: an all-NaN column hard-crashes HistGradientBoosting.

    A grid snapshot job that stops must cost the model one feature, not take the
    nightly velocity retrain down with it.
    """
    f = _toy()
    f["grid_discount"] = np.nan
    f["grid_age_days"] = np.nan
    m = V.VelocityModel().fit(f)
    assert set(m.dropped_features_) == {"grid_discount", "grid_age_days"}
    assert "grid_discount" not in m.features_
    assert m.predict_days(f)["expected_days"].notna().any()


def test_unseen_category_does_not_crash_prediction():
    f = _toy()
    m = V.VelocityModel().fit(f)
    novel = f.head(5).copy()
    novel["shape"] = "Trilliant"                    # never in training
    out = m.predict_days(novel)
    assert len(out) == 5


def test_velocity_score_is_relative_to_this_desk():
    """100 = fastest goods THIS desk trades, not a number invented here."""
    f = _toy()
    m = V.VelocityModel().fit(f)
    d = m.predict_days(f)["own_velocity_score"]
    assert d.between(0, 100).all()
    assert d.max() > 80 and d.min() < 20            # it spreads over the book


def test_config_is_the_one_that_ships():
    """Mirrors `retrain.serving_config` — a knob outside it cannot reach a price."""
    a, b = V.serving_velocity_config(), V.serving_velocity_config()
    assert a == b
    assert V.VelocityModel().cfg == V.serving_velocity_config()


def test_market_depth_is_not_a_covariate():
    """MOU 5.2/8.1: the two numbers are never merged, so depth never trains."""
    assert "market_depth" not in V.usable_covariates(_toy())[0]
    assert "market_depth" not in S.COVARIATES


# ---------------------------------------------------------------------------
# the C-index
# ---------------------------------------------------------------------------
def test_concordance_perfect_and_reversed():
    t = [10, 20, 30, 40]
    e = [1, 1, 1, 1]
    assert B.concordance_index(t, e, [-10, -20, -30, -40])[0] == 1.0
    assert B.concordance_index(t, e, [-40, -30, -20, -10])[0] == 0.0


def test_concordance_all_ties_is_one_half():
    c, pairs = B.concordance_index([1, 2, 3], [1, 1, 1], [5, 5, 5])
    assert c == 0.5
    assert pairs == 3


def test_concordance_censoring_rules():
    """Only pairs whose ordering is actually KNOWN may be scored.

    A censored stone that left at day 5 tells us nothing about a stone that sold
    at day 10 — so that pair contributes nothing. Counting it would inflate the
    index for free.
    """
    # i sold at 10, j censored at 5: j's true time is unknown and may exceed 10.
    c, pairs = B.concordance_index([10, 5], [1, 0], [1.0, 0.0])
    assert pairs == 0 and np.isnan(c)
    # i sold at 5, j censored at 10: j had not sold by 5, so the order IS known.
    c, pairs = B.concordance_index([5, 10], [1, 0], [1.0, 0.0])
    assert pairs == 1 and c == 1.0
    # tied times, one sold one censored: the sale precedes the censoring.
    c, pairs = B.concordance_index([7, 7], [1, 0], [1.0, 0.0])
    assert pairs == 1 and c == 1.0
    # tied times, both sold: order genuinely unknown, so not comparable.
    c, pairs = B.concordance_index([7, 7], [1, 1], [1.0, 0.0])
    assert pairs == 0


def test_concordance_matches_a_brute_force_count():
    """The Fenwick tree exists for speed; correctness is checked against O(n^2)."""
    rng = np.random.default_rng(7)
    n = 120
    t = rng.integers(1, 30, n).astype(float)
    e = (rng.random(n) < 0.6).astype(int)
    r = rng.normal(size=n).round(1)          # deliberate rank ties

    conc = tied = comp = 0
    for i in range(n):
        if not e[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if t[j] > t[i] or (t[j] == t[i] and not e[j]):
                comp += 1
                if r[i] > r[j]:
                    conc += 1
                elif r[i] == r[j]:
                    tied += 1
    expected = (conc + 0.5 * tied) / comp
    got, pairs = B.concordance_index(t, e, r)
    assert pairs == comp
    assert got == pytest.approx(expected, abs=1e-12)


def test_concordance_with_nothing_comparable_is_nan_not_a_half():
    c, pairs = B.concordance_index([5, 5], [0, 0], [1.0, 2.0])
    assert pairs == 0 and np.isnan(c)


# ---------------------------------------------------------------------------
# the out-of-time protocol
# ---------------------------------------------------------------------------
def test_training_arm_is_censored_at_the_split():
    """Without this the 'training' set contains its own future and the model reads
    the answer — CLAUDE.md Trap 2 wearing survival clothing."""
    f = _toy()
    tr, te = B.out_of_time_split(f, "2026-03-01")
    split = pd.Timestamp("2026-03-01")
    assert (tr["entered"] < split).all()
    assert (te["entered"] >= split).all()
    # nothing in the training arm may extend past the split
    assert ((tr["entered"] + pd.to_timedelta(tr["duration"], "D")) <= split).all()
    # a stone recorded as sold in train must have sold BEFORE the split
    sold = tr[tr["event"] == 1]
    assert ((sold["entered"] + pd.to_timedelta(sold["duration"], "D")) <= split).all()


def test_split_cannot_see_the_future(records_frame):
    """The real book: no training row may know anything after the split."""
    tr, _ = B.out_of_time_split(records_frame, "2026-06-01")
    split = pd.Timestamp("2026-06-01")
    latest = (tr["entered"] + pd.to_timedelta(tr["duration"], "D")).max()
    assert latest <= split


@pytest.fixture(scope="module")
def records_frame():
    return S.build_survival_frame(grid=None)[0]


def test_beats_the_segment_median_baseline_out_of_time(records_frame):
    """The Phase-B gate itself. The baseline is what `tradeability.py` ships, so
    passing means the model is worth binding the live field to."""
    r = B.evaluate(records_frame)
    assert r["c_index_model"] > r["c_index_segment_median"]
    assert r["c_index_model"] > 0.55            # meaningfully better than a coin
    assert r["comparable_pairs"] > 1000


def test_calibration_table_reports_thin_follow_up_rather_than_guessing(records_frame):
    """A horizon nothing has been followed to gets `None`, never a number."""
    r = B.evaluate(records_frame)
    rows = r["calibration"][90]
    for row in rows:
        if row["n_followed_to_horizon"] == 0:
            assert row["observed"] is None


# ---------------------------------------------------------------------------
# own vs market — the pair that must never merge
# ---------------------------------------------------------------------------
def test_own_and_market_stay_two_numbers_and_a_ratio():
    out = MD.own_vs_market(80, 20)
    assert out["own_velocity_score"] == 80
    assert out["market_depth_score"] == 20
    assert out["velocity_ratio"] == 4.0
    assert "edge" in out["edge"]                 # names the gap explicitly
    # there is deliberately no blended score
    assert not any("combined" in k or "blended" in k or k == "tradeability_score"
                   for k in out)


def test_the_clients_own_nuance_is_surfaced():
    """MOU 5.2: slow market, fast for us -> keep stocking it."""
    assert "edge" in MD.own_vs_market(85, 15)["edge"]
    assert "deep" in MD.own_vs_market(15, 85)["edge"]
    assert "illiquid" in MD.own_vs_market(15, 15)["edge"]


def test_missing_side_is_not_computable_rather_than_defaulted():
    for a, b in ((None, 50), (50, None), (None, None)):
        out = MD.own_vs_market(a, b)
        assert out["velocity_ratio"] is None
        assert out["edge"] is None
        assert "not computable" in out["edge_basis"]


def test_failed_depth_lookup_is_none_never_zero():
    """Zero comps means 'you are alone' and scores WELL; a dead feed must not."""
    assert MD.depth_for("Round", 1.0, "D", "VS1", market=None).depth is None
    assert MD.depth_score(None) is None
    assert MD.depth_score(0) == 0

    class _Boom:
        def comparables(self, *a, **k):
            raise RuntimeError("timeout")

    r = MD.depth_for("Round", 1.0, "D", "VS1", market=_Boom())
    assert r.depth is None and r.score is None
    assert "unavailable" in r.basis


def test_depth_score_is_monotone_and_bounded():
    prev = -1
    for n in (0, 1, 5, 25, 100, 400, 5000):
        s = MD.depth_score(n)
        assert 0 <= s <= 100 and s >= prev
        prev = s


def test_depth_table_states_its_coverage():
    """A partially-covered book must say so, not read as 'thin everywhere'."""
    t = MD.DepthTable(n_requested=10, n_resolved=4, n_failed=6)
    assert t.coverage == 0.4
    assert "4/10" in t.summary()
    assert t.get("never-looked-up").depth is None
