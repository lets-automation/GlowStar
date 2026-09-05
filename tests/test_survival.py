"""Phase-A gate for Workstream B: the censored, left-truncated survival frame.

Every velocity, tradeability and repricing number the desk will read is built on
this frame. The three corrections it makes are exactly the three that are easy
to get silently wrong, so each one is pinned here rather than described in a
docstring:

  (a) Stock is CENSORED, not "never sells";
  (b) the left-truncation guard is applied — and both biases are corrected, or
      neither (MOU 5.4);
  (c) `Ageing` reconciles with the clock the durations are built from.

Plus the leakage guard on `Discount`, which is the trap that does not look like
one, and an equality check against the live FrontOffice field so this module
cannot quietly move a number the client's screen is already bound to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glowstar.data.loaders import load_records
from glowstar.inventory import survival as S


@pytest.fixture(scope="module")
def records():
    return load_records()[0]


@pytest.fixture(scope="module")
def frame(records):
    # grid=None: the grid join is a per-row lookup over 30k+ stones and is
    # covered separately. Nothing asserted here depends on it.
    return S.build_survival_frame(records, grid=None)


# ---------------------------------------------------------------------------
# (a) censoring
# ---------------------------------------------------------------------------
def test_stock_is_censored_not_never_sells(frame):
    f, rep = frame
    stock = f[f["Status"] == "Stock"]
    assert len(stock) > 0
    assert (stock["event"] == 0).all()             # censored...
    assert stock["duration"].gt(0).mean() > 0.9    # ...with a real elapsed time
    assert np.isfinite(stock["duration"]).all()    # never "infinite days to sell"
    assert (f[f["Status"] == "Sold"]["event"] == 1).all()
    assert rep.n_events + rep.n_censored == rep.n_rows


def test_ignoring_censoring_makes_the_book_look_faster(frame):
    """The bias has a DIRECTION, and it is the one the docstring claims.

    A sold-only median cannot see the stones that have not sold yet, so it must
    come out optimistic. If this ever flips, the censoring is not doing anything.
    """
    f, _ = frame
    sold_only = f.loc[f["event"] == 1, "duration"].median()
    corrected = S.km_median(f["duration"], f["event"].astype(bool))
    assert corrected > sold_only


# ---------------------------------------------------------------------------
# (b) left-truncation
# ---------------------------------------------------------------------------
def test_left_truncation_guard_is_applied(frame):
    f, rep = frame
    assert rep.window_start is not None
    assert (f["entered"] >= rep.window_start).all()
    assert rep.n_dropped_left_truncated > 0        # this book really does have them


def test_ignoring_left_truncation_makes_the_book_look_slower(records):
    """The opposite-direction bias, measured — not asserted from the docstring.

    Survivors listed before the sales window are kept while their contemporaries
    that entered AND SOLD back then are absent from the records entirely. So
    including them must inflate the median. Correcting one bias and not the
    other is worse than correcting neither (MOU 5.4).
    """
    guarded, _ = S.build_survival_frame(records, grid=None)
    m_guarded = S.km_median(guarded["duration"], guarded["event"].astype(bool))

    # Rebuild WITHOUT the guard, the same way build_survival_frame would.
    snap = S.observation_asof(records)
    sold = records[records["Status"] == "Sold"].copy()
    sold["dur"] = (sold["OrderDate_dt"] - sold["MarketSheetDate_dt"]).dt.days
    sold["obs"] = True
    stock = records[records["Status"] == "Stock"].copy()
    stock["dur"] = (snap - stock["MarketSheetDate_dt"]).dt.days
    stock["obs"] = False
    both = pd.concat([sold, stock], ignore_index=True)
    both = both[both["dur"].between(0, S.MAX_DURATION_DAYS)]
    m_unguarded = S.km_median(both["dur"], both["obs"])

    assert m_unguarded > m_guarded


# ---------------------------------------------------------------------------
# (c) the clock
# ---------------------------------------------------------------------------
def test_duration_reconciles_with_the_clients_own_ageing(frame, records):
    """`Ageing` IS the client's clock; our duration must BE it, not resemble it.

    Checked on both arms, because they are built from different formulas
    (OrderDate - MarketSheetDate vs snapshot - MarketSheetDate) and only the
    identity proves the second one picked the right snapshot date.
    """
    f, _ = frame
    ages = (records.set_index(records["StoneId"].astype(str))["Ageing"]
            .pipe(pd.to_numeric, errors="coerce"))
    ages = ages[~ages.index.duplicated(keep="last")]
    for status in ("Sold", "Stock"):
        arm = f[f["Status"] == status]
        joined = arm.join(ages.rename("ageing"), on="StoneId")
        ok = (joined["duration"] - joined["ageing"]).abs().le(0)
        assert ok.mean() == 1.0, f"{status}: duration != Ageing on {(~ok).sum()} rows"


def test_available_days_is_not_the_clock(records):
    """Guards against a future session 'simplifying' onto the wrong field."""
    d = records[records["Status"] == "Sold"]
    real = (d["OrderDate_dt"] - d["MarketSheetDate_dt"]).dt.days
    agrees = (pd.to_numeric(d["AvailableDays"], errors="coerce") - real).abs().eq(0).mean()
    assert agrees < 0.9      # it is a DIFFERENT quantity; measured ~0.37


def test_observation_asof_comes_from_the_data_not_the_wall_clock(records):
    """A stale feed must show up as an old date, not as a slower desk.

    Censoring stock at `today` when the snapshot is three days old adds three
    phantom unsold days to every stone in the book.
    """
    snap = S.observation_asof(records)
    stock = records[records["Status"] == "Stock"]
    implied = (pd.to_numeric(stock["Ageing"], errors="coerce")
               - (snap - stock["MarketSheetDate_dt"]).dt.days)
    assert implied.abs().eq(0).mean() == 1.0

    # And a doctored frame must move the date, not silently keep "now".
    older = records.copy()
    mask = older["Status"] == "Stock"
    older.loc[mask, "MarketSheetDate_dt"] = older.loc[mask, "MarketSheetDate_dt"] - pd.Timedelta(days=5)
    assert S.observation_asof(older) == snap - pd.Timedelta(days=5)


# ---------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------
def test_discount_is_never_a_velocity_covariate():
    """`Discount` is overwritten with the realized price on a Sold row.

    It therefore means "asking" in the censored arm and "closing" in the event
    arm; a model given it learns to read the answer. `BasePriceDiscount` is the
    stable listing-time field and is what the frame carries instead.
    """
    assert "Discount" not in S.COVARIATES
    assert "Discount" in S.FORBIDDEN_VELOCITY_FEATURES
    assert "base_discount" in S.COVARIATES
    assert not (S.FORBIDDEN_VELOCITY_FEATURES & set(S.COVARIATES))


def test_discount_really_does_leak_on_this_book(records):
    """The measurement behind the rule above, re-run rather than cited.

    If the client's feed ever stops overwriting `Discount` at sale this test
    fails and the rule can be revisited on evidence — which is the only way it
    should ever be revisited.
    """
    sold = records[records["Status"] == "Sold"]
    same = (pd.to_numeric(sold["Discount"], errors="coerce")
            == pd.to_numeric(sold["FDiscount"], errors="coerce")).mean()
    assert same > 0.8, "Discount no longer tracks the final price — re-measure the rule"


def test_every_column_is_either_a_covariate_or_declared_bookkeeping(frame):
    """No third category — that is where a leaked field would hide.

    `Status`, `duration` and `event` ARE post-listing facts and are carried on
    purpose, so the frame states which columns a model may see and which exist
    only to join and report. A column in neither list fails the build.
    """
    f, _ = frame
    assert set(f.columns) == set(S.COVARIATES) | set(S.BOOKKEEPING_COLUMNS)
    modelled = set(S.COVARIATES) & S.FORBIDDEN_VELOCITY_FEATURES
    assert not modelled, f"a model can see post-listing fields: {sorted(modelled)}"


def test_tinge_uses_the_unassessed_sentinel_never_nan_or_zero(frame):
    """CLAUDE.md Trap 4: 0.0 means assessed-and-clean; NaN would crash the GBM."""
    f, _ = frame
    for c in ("brown_ord", "milky_ord", "shade_ord", "green_ord"):
        assert f[c].notna().all(), f"{c} carries NaN — HistGradientBoosting will not fit"
        assert (f[c] == -1.0).any() or (f[c] >= 0).all()


# ---------------------------------------------------------------------------
# Kaplan-Meier mechanics
# ---------------------------------------------------------------------------
def test_km_matches_a_hand_worked_example():
    """Textbook check: 6 subjects, one censored, worked by hand.

    times  : 1, 2+, 3, 4, 5, 6   ('+' = censored)
    t=1: 6 at risk, 1 event -> S = 5/6           = 0.8333
    t=3: 4 at risk, 1 event -> S = 0.8333 * 3/4  = 0.6250
    t=4: 3 at risk, 1 event -> S = 0.6250 * 2/3  = 0.4167
    """
    c = S.km_curve([1, 2, 3, 4, 5, 6], [True, False, True, True, True, True])
    assert list(c.times) == [1.0, 3.0, 4.0, 5.0, 6.0]
    assert c.survival[0] == pytest.approx(5 / 6, abs=1e-9)
    assert c.survival[1] == pytest.approx(0.625, abs=1e-9)
    assert c.survival[2] == pytest.approx(0.4166666, abs=1e-6)
    assert c.median() == 4.0                    # first time S drops to <= 0.5
    assert c.n_events == 5 and c.n_censored == 1


def test_km_median_not_reached_is_infinity_never_a_guess():
    """MOU 10.3: 'not reached' is the honest answer for a genuinely slow segment."""
    c = S.km_curve([10, 20, 30, 40], [True, False, False, False])
    assert c.median() == float("inf")
    assert c.survival[-1] > 0.5


def test_greenwood_band_stays_inside_zero_one_and_brackets_the_curve():
    rng = np.random.default_rng(0)
    d = rng.integers(1, 120, 300)
    o = rng.random(300) < 0.7
    c = S.km_curve(d, o)
    assert (c.lower >= 0.0).all() and (c.upper <= 1.0).all()
    assert (c.lower <= c.survival + 1e-12).all()
    assert (c.upper >= c.survival - 1e-12).all()


def test_median_ci_runs_forwards_and_contains_the_median():
    """This ran BACKWARDS on the first build (printed '35-33 days')."""
    rng = np.random.default_rng(1)
    d = rng.integers(1, 200, 500)
    o = rng.random(500) < 0.8
    c = S.km_curve(d, o)
    lo, hi = c.median_ci()
    m = c.median()
    assert lo <= m <= hi
    assert lo <= hi


def test_thin_segments_get_a_wide_band_not_a_confident_one():
    """A wide interval is a correct answer, not a defect (MOU 10.3)."""
    thin = S.km_curve([5, 40, 90], [True, False, False])
    wide_lo, wide_hi = thin.median_ci()
    assert wide_hi == float("inf")


def test_empty_input_does_not_raise():
    c = S.km_curve([], [])
    assert c.median() == float("inf")
    assert c.n_events == 0


# ---------------------------------------------------------------------------
# the live FrontOffice field must not move
# ---------------------------------------------------------------------------
def test_agrees_with_the_shipped_tradeability_table(frame):
    """MOU 2.2: the desk's screen is already bound to these days-to-sell numbers.

    The general frame must reproduce the shipped table EXACTLY on the shipped
    segmentation, so replacing the hand-rolled estimator with this one moves
    nothing the client is reading. Measured at build time: 236/236 segments,
    max difference 0.0 days.
    """
    from glowstar.service import tradeability as T

    f, _ = frame
    t = T.build_table(force=True)
    seg = f["shape"] + "|" + f["color"] + "|" + f["clarity"]
    mine = {}
    for name, g in f.groupby(seg):
        if len(g) < T.MIN_SEGMENT:
            continue
        m = S.km_median(g["duration"], g["event"].astype(bool))
        if np.isfinite(m):
            mine[name] = m
    common = set(mine) & set(t.by_segment)
    assert len(common) > 50, "the two tables barely overlap — something diverged"
    worst = max(abs(mine[s] - t.by_segment[s]) for s in common)
    assert worst == 0.0, f"days-to-sell moved by up to {worst} days on the live field"
