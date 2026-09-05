"""Per-stone expected days-to-sell — Workstream B, Phase B.

WHY A DISCRETE-TIME HAZARD AND NOT COX / RANDOM SURVIVAL FOREST
---------------------------------------------------------------
Measured 2026-08-24 and re-confirmed here; this is not a preference:

  * `lifelines` is BLOCKED. It caps `pandas<3.0`, so installing it downgrades the
    project's pandas across the whole venv and takes the pricing engine, the
    loaders and every saved model with it.
  * `scikit-survival` is dependency-clean but WILL NOT BUILD on this box: its
    dependency `ecos` publishes no wheel for Python 3.14 and there is no C
    compiler here (`cl.exe` and `gcc` both absent).

So the model is built on the stack that is already pinned and already trusted:
each stone is expanded into (stone, period) rows carrying "did it sell in this
period?", and the repo's own HistGradientBoosting classifier fits the hazard.
Censoring is handled natively — a censored stone simply contributes rows for the
periods it fully survived — and because the period index is itself a feature,
the baseline hazard is free to change shape over time. That is a strictly weaker
assumption than Cox's proportional hazards, not a compromise.

The output is a whole survival curve per stone, so days-to-sell AND its interval
both fall out of the same object rather than being bolted on.

WHAT IS DELIBERATELY *NOT* IN THIS MODEL
-----------------------------------------
**Market depth is not a covariate.** MOU 5.2 and 8.1: own velocity and market
liquidity are two numbers and a ratio, never one blended score, because the gap
between them is the client's edge and is the reason this workstream exists. A
model with depth baked in is no longer "how fast do WE sell it".

There is a second, independent reason. Depth is only observable NOW — there is
no banked history of it — so a training row for a January sale would carry
August's depth. That measurement error is not random: it is ~zero for the
censored arm (still in stock today) and grows the further back an event sits.
That is the same shape as the `Discount` trap in `survival.py`, and it is worth
naming because it is the kind of covariate that improves a score while teaching
the model nothing.

Depth is computed in `market_depth.py`, reported beside this model's output, and
`own_vs_market()` puts the two side by side with their ratio.

RECALIBRATION WAS TRIED AND REJECTED — ON MEASUREMENT, NOT TASTE
------------------------------------------------------------------
The obvious next move after seeing the extreme bins spread is an out-of-fold
isotonic map on the hazards: monotone, so it cannot change the C-index, and it
is the textbook fix. Measured on the inner validation split it made calibration
WORSE — weighted |predicted - observed| went 0.0435 -> 0.0775.

The reason is worth keeping: the model over-predicts the fastest bin on the
outer test window and UNDER-predicts across the whole inner validation window.
Those are opposite signs, so any fixed correction fitted on one is noise fitted
backwards for the other. The residual spread is window-to-window variation on
under a year of history, not a stable model defect, and a calibration layer
would launder that noise into something that looks like a correction.

So there is no calibration layer. The backtest prints the calibration table
instead, and a reader can see the spread for themselves.

CALIBRATION AND HONESTY
-----------------------
`predict_days()` returns `expected_days` with an 80% interval (the same coverage
the pricing engine ships, `SETTINGS.interval_coverage`) and a `basis` string on
every row. Where the curve does not reach 0.5 inside the horizon the answer is
`not reached` and `expected_days` is None — never a fabricated point estimate
(MOU 10.3). Out-of-time C-index and calibration:

    python -m glowstar.validation.survival_backtest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import SETTINGS
from . import survival as S

log = logging.getLogger(__name__)

# Period edges in days. Fine early because that is where this book actually
# trades (median sale 19d, p75 48d) and coarse later where the data thins. The
# last edge is the HORIZON: beyond it the honest answer is "not reached inside
# the window", and the window is bounded by the sale history itself (max
# observed duration ~252d), so a wider horizon would be extrapolation.
PERIOD_EDGES: tuple[float, ...] = (
    0, 3, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180, 240, 270,
)
HORIZON_DAYS: float = float(PERIOD_EDGES[-1])


@dataclass
class VelocityConfig:
    """Every knob that changes a published days-to-sell number lives here.

    Mirrors `training.retrain.serving_config()`: the retrain gate scores the
    config that SHIPS, and a knob outside this object cannot reach the desk
    (CLAUDE.md Trap 5).
    """

    period_edges: tuple[float, ...] = PERIOD_EDGES
    learning_rate: float = 0.06
    # TUNED ON AN INNER SPLIT INSIDE THE TRAINING WINDOW (2026-04-15), never on
    # the out-of-time test set — picking these against the number the gate
    # reports would make the gate score its own tuning. Measured there:
    #
    #   max_iter=300 -> C 0.631, weighted calibration error 0.107   overfit
    #   max_iter=120 -> C 0.638, 0.075
    #   max_iter= 60 -> C 0.632, 0.040   <- the knee; C is flat from 20 to 80
    #   max_iter= 20 -> C 0.628, 0.045
    #
    # Fewer iterations cost nothing in ranking and cut the calibration error
    # roughly threefold. That trade only shows up if you look at BOTH numbers:
    # the C-index alone is flat across the whole sweep and would have happily
    # licensed the overfit model.
    max_iter: int = 60
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 300
    l2_regularization: float = 10.0
    random_state: int = 42
    # Interval coverage, shared with the pricing engine so the desk does not
    # have to hold two different meanings of "the band" in their head.
    coverage: float = SETTINGS.interval_coverage
    # A segment with fewer sales than this is not given its own norm; the score
    # backs off to a coarser one and the basis says so.
    min_segment_sales: int = 15

    @property
    def lo_q(self) -> float:
        return (1.0 - self.coverage) / 2.0

    @property
    def hi_q(self) -> float:
        return 1.0 - self.lo_q


def serving_velocity_config() -> VelocityConfig:
    """THE velocity config. Training, the gate and serving all construct it here.

    The pricing engine learned this the hard way: `price_and_report` shipped one
    configuration while the gate and every backtest scored another, and the
    published accuracy was from a pipeline the client never received. Anything
    that changes a days-to-sell number for real constructs its config HERE.
    """
    return VelocityConfig()


# ---------------------------------------------------------------------------
# person-period expansion
# ---------------------------------------------------------------------------
def expand_periods(frame: pd.DataFrame, edges=PERIOD_EDGES) -> pd.DataFrame:
    """One row per (stone, period the stone was at risk in), with EXPOSURE.

    Conventions, chosen so a same-day sale (`duration == 0`, which this book has
    plenty of) is not silently dropped:

      * period k covers [edges[k], edges[k+1]);
      * a stone is AT RISK in k iff `duration >= edges[k]`;
      * a SOLD stone's event lands in the k with `edges[k] <= duration < edges[k+1]`;
      * a CENSORED stone whose observation ends mid-period contributes that
        period with `sold_in_period=0` and `exposure` = the fraction of it we
        actually watched.

    THE EXPOSURE WEIGHT IS NOT COSMETIC — it was measured
    -----------------------------------------------------
    The obvious implementation drops a censored stone's final, partial period
    instead of weighting it. That removes a NON-EVENT from the denominator of
    that period's hazard, so every hazard comes out too high and the error
    compounds across periods. On this book it was worth, on the model's own
    training data:

        day 30:  P(sold) 0.499 predicted vs 0.481 actual
        day 60:  0.693 vs 0.622
        day 90:  0.798 vs 0.697

    The C-index never noticed — a monotone distortion changes no ranking — so
    only the calibration table caught it. Weighting by observed exposure is the
    classical actuarial (life-table) correction, generalised from "subtract half
    the mid-period censorings" to "subtract exactly the part we did not see".
    """
    e = np.asarray(edges, dtype=float)
    n_per = len(e) - 1
    dur = frame["duration"].to_numpy(float)
    ev = frame["event"].to_numpy(int)

    rows, periods, y, expo = [], [], [], []
    for k in range(n_per):
        a, b = e[k], e[k + 1]
        idx = np.nonzero(dur >= a)[0]
        if not len(idx):
            continue
        d, is_event = dur[idx], ev[idx] == 1
        sold_here = is_event & (d >= a) & (d < b)
        # Full exposure unless the stone was CENSORED partway through this
        # period; a sold stone is at risk for the whole period by construction.
        partial = (~is_event) & (d < b)
        w = np.where(partial, (d - a) / (b - a), 1.0)
        keep = w > 0.0                       # a zero-exposure row teaches nothing
        rows.append(idx[keep])
        periods.append(np.full(int(keep.sum()), k))
        y.append(sold_here[keep].astype(int))
        expo.append(w[keep])

    if not rows:
        return frame.head(0).assign(period=pd.Series(dtype=int),
                                    sold_in_period=pd.Series(dtype=int),
                                    exposure=pd.Series(dtype=float))
    idx = np.concatenate(rows)
    out = frame.iloc[idx].copy()
    out["period"] = np.concatenate(periods)
    out["sold_in_period"] = np.concatenate(y)
    out["exposure"] = np.concatenate(expo)
    return out.reset_index(drop=True)


def usable_covariates(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(usable, dropped) — numeric covariates that are not entirely missing.

    An ALL-NaN column hard-crashes HistGradientBoosting at fit ("window shape
    cannot be larger than input array shape"), and CLAUDE.md names this as the
    way a feed that stops sending a field takes the nightly retrain down. It is
    not hypothetical here: build the frame with `grid=None`, or let the daily
    grid snapshot job die, and `grid_discount` / `grid_age_days` are exactly
    that column.

    So a dead feed DEGRADES the model — one feature poorer, and the drop is
    logged and carried on the model card — instead of killing the job. Tinge
    columns can never land here: `survival._tinge` fills them with the
    UNASSESSED sentinel rather than NaN, for this same reason.
    """
    usable, dropped = [], []
    for c in S.COVARIATES:
        if c in S.CATEGORICAL_COVARIATES:
            usable.append(c)
            continue
        if pd.to_numeric(frame[c], errors="coerce").notna().any():
            usable.append(c)
        else:
            dropped.append(c)
    return usable, dropped


def _design(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The model matrix: the chosen covariates + the period index, nothing else.

    Reads an explicit feature list rather than "every column that looks
    numeric", so a column added to the frame for reporting can never become a
    feature by accident.
    """
    x = df[list(features)].copy()
    for c in features:
        if c in S.CATEGORICAL_COVARIATES:
            x[c] = x[c].astype("category")
    # The period index as an ordinary numeric feature: the tree can bend the
    # baseline hazard into any shape, so proportional hazards is never assumed.
    x["period"] = df["period"].astype(float)
    return x


@dataclass
class VelocityModel:
    """Discrete-time hazard model over the survival frame."""

    cfg: VelocityConfig = field(default_factory=serving_velocity_config)
    model: object | None = None
    categories_: dict[str, pd.Index] = field(default_factory=dict)
    features_: list[str] = field(default_factory=list)
    dropped_features_: list[str] = field(default_factory=list)
    n_train_stones: int = 0
    n_train_rows: int = 0
    n_train_events: int = 0
    trained_at: str | None = None
    # The client's own days-to-sell distribution, for scoring RELATIVE to how
    # this desk actually trades rather than to a number invented here.
    score_reference_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "VelocityModel":
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.features_, self.dropped_features_ = usable_covariates(frame)
        if self.dropped_features_:
            log.warning("Velocity model: dropping all-missing covariates %s — the "
                        "feed behind them is not arriving. The model still fits, "
                        "one feature poorer; fix the feed.", self.dropped_features_)
        long = expand_periods(frame, self.cfg.period_edges)
        x = _design(long, self.features_)
        self.categories_ = {c: x[c].cat.categories
                            for c in self.features_ if c in S.CATEGORICAL_COVARIATES}
        y = long["sold_in_period"].to_numpy(int)

        self.model = HistGradientBoostingClassifier(
            learning_rate=self.cfg.learning_rate,
            max_iter=self.cfg.max_iter,
            max_leaf_nodes=self.cfg.max_leaf_nodes,
            min_samples_leaf=self.cfg.min_samples_leaf,
            l2_regularization=self.cfg.l2_regularization,
            categorical_features="from_dtype",
            random_state=self.cfg.random_state,
        ).fit(x, y, sample_weight=long["exposure"].to_numpy(float))

        self.n_train_stones = int(len(frame))
        self.n_train_rows = int(len(long))
        self.n_train_events = int(y.sum())
        self.trained_at = pd.Timestamp.now().isoformat(timespec="seconds")

        # Score reference: the in-sample distribution of predicted days over the
        # training book. A stone scores against how this desk trades, exactly as
        # `tradeability.py` takes its cutoffs from the client's own quintiles.
        ref = self._median_days(self.predict_survival(frame))
        self.score_reference_ = np.sort(ref[np.isfinite(ref)])
        log.info("Velocity model: %d stones -> %d person-period rows, %d events, "
                 "horizon %.0fd", self.n_train_stones, self.n_train_rows,
                 self.n_train_events, HORIZON_DAYS)
        return self

    # -- prediction ---------------------------------------------------------
    def predict_hazards(self, frame: pd.DataFrame) -> np.ndarray:
        """(n_stones, n_periods) matrix of per-period sell probabilities."""
        if self.model is None:
            raise RuntimeError("VelocityModel is not fitted")
        n_per = len(self.cfg.period_edges) - 1
        n = len(frame)
        haz = np.zeros((n, n_per), dtype=float)
        base = frame[list(self.features_)].copy()
        for k in range(n_per):
            block = base.copy()
            block["period"] = float(k)
            for c, cats in self.categories_.items():
                # Pin the training categories: a shape or colour the model never
                # saw becomes NaN (which the GBM handles) rather than silently
                # re-coding every category to a different integer — which would
                # give a Trilliant whatever the model learned about Rounds.
                # Unseen values are mapped to NaN explicitly rather than left to
                # the Categorical constructor, which now deprecates doing it.
                vals = block[c].where(block[c].isin(cats))
                block[c] = pd.Categorical(vals, categories=cats)
            haz[:, k] = self.model.predict_proba(block)[:, 1]
        return haz

    def predict_survival(self, frame: pd.DataFrame) -> np.ndarray:
        """(n_stones, n_periods) survival S(t) at each period BOUNDARY."""
        return np.cumprod(1.0 - self.predict_hazards(frame), axis=1)

    def _crossing(self, surv: np.ndarray, level: float) -> np.ndarray:
        """Days at which each stone's curve first falls to `level`.

        Linearly interpolated INSIDE the period it crosses in, so two stones in
        the same period do not collapse onto one identical answer — which is
        what makes a 0-100 score out of this usable at all. `inf` when the curve
        never gets there inside the horizon.
        """
        e = np.asarray(self.cfg.period_edges, dtype=float)
        out = np.full(len(surv), np.inf)
        prev = np.ones(len(surv))
        for k in range(surv.shape[1]):
            s = surv[:, k]
            hit = (out == np.inf) & (s <= level)
            if hit.any():
                p, c = prev[hit], s[hit]
                span = np.where(p > c, (p - level) / np.maximum(p - c, 1e-12), 0.0)
                out[hit] = e[k] + span * (e[k + 1] - e[k])
            prev = s
        return out

    def _median_days(self, surv: np.ndarray) -> np.ndarray:
        return self._crossing(surv, 0.5)

    def predict_days(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Expected days-to-sell per stone, with its interval and its basis.

        `expected_days` is None where the curve does not reach 0.5 inside the
        horizon: that segment genuinely does not turn over that far in the
        history we have, and inventing a number for it is the one thing MOU 10.3
        forbids outright.
        """
        surv = self.predict_survival(frame)
        med = self._median_days(surv)
        lo = self._crossing(surv, 1.0 - self.cfg.lo_q)     # optimistic end
        hi = self._crossing(surv, 1.0 - self.cfg.hi_q)     # pessimistic end

        finite = np.isfinite(med)
        out = pd.DataFrame({
            "StoneId": frame["StoneId"].to_numpy(),
            "segment": frame["segment"].to_numpy(),
            "expected_days": np.where(finite, np.round(med, 1), np.nan),
            "days_low": np.where(np.isfinite(lo), np.round(lo, 1), np.nan),
            "days_high": np.where(np.isfinite(hi), np.round(hi, 1), np.nan),
            "reached_median": finite,
            "p_sold_30d": np.round(1.0 - self._survival_at(surv, 30.0), 3),
            "p_sold_90d": np.round(1.0 - self._survival_at(surv, 90.0), 3),
        })
        out["own_velocity_score"] = self.velocity_score(med)
        out["basis"] = [
            (f"discrete-time hazard, {self.n_train_events:,} sales over "
             f"{self.n_train_stones:,} listings"
             + ("" if f_ else f"; median not reached inside {HORIZON_DAYS:.0f}d"))
            for f_ in finite
        ]
        return out

    def predict_remaining_days(self, frame: pd.DataFrame,
                               age_col: str = "duration") -> pd.DataFrame:
        """Days-to-sell for a stone that has ALREADY sat unsold for `age` days.

        This is what the desk actually needs on a stock book, and it is what
        makes "a stone past its segment median is slowing" a computation instead
        of a rule of thumb. Conditioning is exact rather than heuristic:

            P(sells by t | still here at age) = 1 - S(t) / S(age)

        A 120-day-old stone in a 40-day segment is not simply "80 days late" —
        it has demonstrated that it is not an average stone in that segment, and
        the conditional curve says how much longer it is really likely to take.

        Adds `age_days`, `expected_total_days` (from listing) and
        `expected_remaining_days` (from today). Both are reported because they
        answer different questions: the total is comparable across the book, the
        remaining is what the desk decides on.
        """
        surv = self.predict_survival(frame)
        age = frame[age_col].to_numpy(float)
        s_age = np.array([self._survival_at(surv[i:i + 1], a)[0]
                          for i, a in enumerate(age)])
        # Renormalise onto "still here at `age`", then floor at the age itself:
        # a stone cannot sell in the past.
        cond = np.clip(surv / np.maximum(s_age[:, None], 1e-9), 0.0, 1.0)
        e = np.asarray(self.cfg.period_edges, dtype=float)
        cond = np.where(e[None, 1:] <= age[:, None], 1.0, cond)

        total = self._crossing(cond, 0.5)
        lo = self._crossing(cond, 1.0 - self.cfg.lo_q)
        hi = self._crossing(cond, 1.0 - self.cfg.hi_q)
        finite = np.isfinite(total)
        out = pd.DataFrame({
            "StoneId": frame["StoneId"].to_numpy(),
            "segment": frame["segment"].to_numpy(),
            "age_days": np.round(age, 0),
            "expected_total_days": np.where(finite, np.round(total, 1), np.nan),
            "expected_remaining_days": np.where(
                finite, np.round(np.maximum(total - age, 0.0), 1), np.nan),
            "remaining_low": np.where(np.isfinite(lo), np.round(np.maximum(lo - age, 0.0), 1), np.nan),
            "remaining_high": np.where(np.isfinite(hi), np.round(np.maximum(hi - age, 0.0), 1), np.nan),
            "reached_median": finite,
        })
        # A stone already old enough that its conditional median crowds the
        # horizon has an answer TRUNCATED by the window, not measured by it —
        # the estimate is a floor. Without this flag the oldest ageing bucket
        # reads as if it were turning faster than the bucket below it, which is
        # purely the horizon biting.
        horizon = float(self.cfg.period_edges[-1])
        out["horizon_limited"] = (~finite) | (total >= 0.9 * horizon)
        # SCORE THE REMAINING DAYS, NOT THE TOTAL. The reference distribution is
        # "days from listing to sale", i.e. days from the moment of assessment —
        # so for a stone being assessed today the comparable quantity is how
        # much longer it will take, not how long it will have taken in total.
        #
        # Scoring the total instead charges a stone for time already elapsed AND
        # for its expected wait, and stock is by construction the surviving
        # (slower) tail. It put 59% of the live book in "Slow" and scored a
        # stone expected to sell in 35 more days at 0/100. The benchmark check
        # below is what caught it.
        out["own_velocity_score"] = self.velocity_score(
            out["expected_remaining_days"].to_numpy(float))
        out["basis"] = [
            (f"conditional on {a:.0f} days already unsold; discrete-time hazard over "
             f"{self.n_train_events:,} sales"
             + ("" if f_ else f"; median not reached inside {horizon:.0f}d")
             + ("; estimate truncated by the observation window — read it as a floor"
                if h_ and f_ else ""))
            for a, f_, h_ in zip(age, finite, out["horizon_limited"])
        ]
        return out

    def _survival_at(self, surv: np.ndarray, day: float) -> np.ndarray:
        """P(still unsold at `day`) — the last period boundary AT OR BEFORE it.

        `surv[:, k]` is survival at the END of period k, i.e. at
        `period_edges[k+1]`, so the column for `day` is the largest k with
        `edges[k+1] <= day` — NOT the period that contains `day`. Taking the
        containing period reads survival at the period's far edge and reports
        P(sold by day 30) as if it were P(sold by day 45): the first build did
        exactly that and predicted 0.99 against an observed 0.58. It was
        invisible in the C-index (a monotone distortion does not change any
        ranking) and only the calibration table caught it.
        """
        e = np.asarray(self.cfg.period_edges, dtype=float)
        k = int(np.searchsorted(e, day, side="right") - 2)
        if k < 0:
            return np.ones(len(surv))
        return surv[:, min(k, surv.shape[1] - 1)]

    def velocity_score(self, days: np.ndarray) -> np.ndarray:
        """0-100, where 100 is the fastest goods THIS DESK trades.

        A percentile against the client's own distribution, not against an
        invented day count: "slow" means slow for Glow Star. A stone whose
        median is never reached scores 0 — it is the slowest thing in the book
        by definition, which is a real answer rather than a missing one.
        """
        ref = self.score_reference_
        d = np.asarray(days, dtype=float)
        if ref is None or not len(ref):
            return np.full(len(d), np.nan)
        rank = np.searchsorted(ref, d, side="left") / len(ref)
        score = np.round(100.0 * (1.0 - rank), 0)
        return np.where(np.isfinite(d), score, 0.0)


# ---------------------------------------------------------------------------
# the segment-median baseline the model has to beat
# ---------------------------------------------------------------------------
@dataclass
class SegmentMedianBaseline:
    """KM median per segment, with hierarchical backoff. The bar for Phase B.

    This is not a straw man: it is essentially what `service/tradeability.py`
    ships today, so "the model beats the baseline" means "the model is worth
    replacing the live field with". Backoff runs along `market.segments`'
    hierarchy so every stone gets an answer and the level it came from is
    reported, never disguised.
    """

    min_sales: int = 15
    levels_: list[dict] = field(default_factory=list)
    overall_: float = float("nan")

    _KEYS = (
        ("shape", "size_band", "color", "clarity"),
        ("shape", "size_band", "color"),
        ("shape", "size_band"),
        ("shape",),
    )

    def fit(self, frame: pd.DataFrame) -> "SegmentMedianBaseline":
        self.levels_ = []
        for keys in self._KEYS:
            table: dict = {}
            for name, g in frame.groupby(list(keys), observed=True):
                if int(g["event"].sum()) < self.min_sales:
                    continue
                m = S.km_median(g["duration"], g["event"].astype(bool))
                if np.isfinite(m):
                    table[name if isinstance(name, tuple) else (name,)] = float(m)
            self.levels_.append({"keys": keys, "table": table})
        m = S.km_median(frame["duration"], frame["event"].astype(bool))
        self.overall_ = float(m) if np.isfinite(m) else float(
            frame.loc[frame.event == 1, "duration"].median())
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        days = np.full(len(frame), np.nan)
        basis = ["whole book (no segment support)"] * len(frame)
        cols = {c: frame[c].to_numpy() for c in
                ("shape", "size_band", "color", "clarity")}
        for lvl in self.levels_:
            keys, table = lvl["keys"], lvl["table"]
            if not table:
                continue
            tup = list(zip(*[cols[k] for k in keys]))
            for i, t in enumerate(tup):
                if np.isnan(days[i]) and t in table:
                    days[i] = table[t]
                    basis[i] = "segment " + "|".join(str(v) for v in t)
        days = np.where(np.isnan(days), self.overall_, days)
        return days, basis


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frame, rep = S.build_survival_frame()
    print(rep.summary())
    m = VelocityModel().fit(frame)
    days = m.predict_days(frame)
    print(days[["expected_days", "days_low", "days_high",
                "own_velocity_score"]].describe().round(1).to_string())
    print(f"\nmedian not reached inside {HORIZON_DAYS:.0f}d for "
          f"{(~days.reached_median).mean():.1%} of the book")


if __name__ == "__main__":
    main()
