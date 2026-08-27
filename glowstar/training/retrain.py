"""Nightly retrain with an accuracy PROMOTION GATE (brief Sections 12, 13).

Pipeline:
  1. Bank a fresh live snapshot (when credentials are set), else use banked/file.
  2. Assemble the full sold history by UNIONing every banked snapshot
     (glowstar/data/history.py) — this is what grows the trainable window.
  3. Train a CANDIDATE on the out-of-time train split and measure it on the most
     recent window (leakage-free, the brief's honest accuracy harness).
  4. GATE: promote the candidate to "current" only if it matches or beats the
     incumbent within tolerance. A worse model is saved (for audit) but NOT
     promoted — serving keeps the incumbent. This makes "retrain after every X
     time" safe: a bad data day can never silently degrade live pricing.
  5. The promoted artifact is a PRODUCTION engine trained on ALL sold history +
     human feedback (not just the train split).

Schedule daily (after the snapshot pull). Windows Task Scheduler / cron lines are
in the README. Run once:  python -m glowstar.training.retrain
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import numpy as np

from ..config import SETTINGS
from ..feedback import store as fbstore
from ..models.engine import PricingEngine, EngineConfig
from ..models import registry
from ..validation.backtest import time_split
from ..validation import metrics as M

log = logging.getLogger(__name__)

# THE GATE'S HORIZON, in days — how far ahead the candidate must predict.
#
# 7, because that is what production faces. The gate used to train on everything
# before a FIXED split (2026-06-01) and score every sale after it, i.e. a model
# frozen in June predicting up to twelve weeks ahead. Production retrains EVERY
# NIGHT and prices the next day. Those are different questions and they do not
# even rank changes the same way:
#
#     forward-drift correction OFF:  7-day horizon 1.83  |  12-week horizon 2.59
#     forward-drift correction ON :  7-day horizon 1.99  |  12-week horizon 2.33
#
# A staleness correction helps a stale model and hurts a fresh one, so the two
# protocols disagree about which model is better. Scoring the long horizon made
# the gate reject the configuration that was measurably BETTER for the client
# (realized MAE on actual sales: 2.48 -> 1.44 the week it shipped).
#
# CLAUDE.md Trap 5, one more time: measure the path that SHIPS.
GATE_HORIZON_DAYS = 7

# Identifies WHICH question `test_mae` answers, stamped on every card. Bump this
# whenever the protocol changes, so `best_mae()` cannot compare across the change
# — that near-miss is documented in registry.best_mae.
METRIC_PROTOCOL = f"rolling{GATE_HORIZON_DAYS}d.v1"


def gate_split(sold, horizon_days: int = GATE_HORIZON_DAYS):
    """Train on everything up to `horizon_days` ago; test on that window.

    Mirrors production: a model fitted last night pricing today's stones. Returns
    (train, test, origin).
    """
    import pandas as _pd
    d = sold.copy()
    d["OrderDate_dt"] = _pd.to_datetime(d["OrderDate_dt"])
    origin = d["OrderDate_dt"].max() - _pd.Timedelta(days=horizon_days)
    return d[d["OrderDate_dt"] < origin], d[d["OrderDate_dt"] >= origin], origin


# A candidate may be at most this many MAE points worse than the INCUMBENT and
# still be promoted (small wiggle for run-to-run noise). Configurable.
PROMOTE_TOLERANCE_PTS = 0.25

# ...and at most this many points worse than the BEST model ever recorded.
#
# WHY THIS SECOND BOUND EXISTS — a real, observed failure
# ------------------------------------------------------
# The incumbent-only rule is a RATCHET. The reference point is replaced by every
# promotion, so the bar it sets moves with it. A candidate 0.24 worse than the
# incumbent is promoted, becomes the incumbent, and the next night's bar is 0.24
# higher again. Nothing bounds the total.
#
# This is not hypothetical. Measured on the first three production models:
#     2.469 -> 2.605 -> 2.815   (+0.346 in two nights)
# Both promotions were inside tolerance; the gate never objected. Projected
# forward at the same rate the model reaches MAE ~4.96 in ten nights, with every
# single log line reading `promoted: True`.
#
# So drift is now measured from the BEST MAE ever achieved, not from whatever
# happened to be live yesterday.
#
# WHY A BOUND AND NOT "must beat the best"
# ----------------------------------------
# Refusing anything worse than the best-ever would freeze the model the first
# time the market genuinely gets harder or the test window shifts — and a frozen
# model that quietly stops tracking reality is a failure this project has already
# been bitten by (see the feedback note below). Bounding cumulative drift keeps
# the model tracking day to day while capping how far it can wander.
MAX_DRIFT_FROM_BEST_PTS = 0.50

# Absolute backstop. Deliberately far above any value this engine has ever
# produced (it has run between ~2.4 and ~4.0), because its job is to catch a
# CATASTROPHE — a broken feed, a corrupted target, a bug that makes the model
# meaningless — not to act as a quality bar. The relative rules above do the
# quality work. Set too low it would block legitimate models during a genuinely
# hard market, which is the "gate rejects everything and the retrain silently
# freezes" failure this project has already been bitten by.
MAX_ACCEPTABLE_MAE = 8.0


def gate_decision(cand_mae: float | None, inc_mae: float | None,
                  best_mae: float | None = None,
                  tolerance: float = PROMOTE_TOLERANCE_PTS,
                  max_drift: float = MAX_DRIFT_FROM_BEST_PTS,
                  ceiling: float = MAX_ACCEPTABLE_MAE) -> tuple[bool, str]:
    """Pure promotion rule (unit-testable). ALL of these must hold:

      1. the candidate is not materially worse than the INCUMBENT (day-to-day), and
      2. it has not drifted too far from the BEST model ever seen (anti-ratchet), and
      3. it is under an absolute ceiling (catastrophe backstop).

    `best_mae` is optional so old callers keep working, but the nightly retrain
    always passes it — without it, rule 2 cannot apply and the ratchet returns.
    """
    if cand_mae is None:
        return False, "no test window to evaluate — not promoting"

    if cand_mae > ceiling:
        return False, (f"candidate MAE {cand_mae} exceeds the absolute ceiling "
                       f"{ceiling} — refusing regardless of the incumbent")

    if inc_mae is None:
        return True, "no incumbent — promoting first model"

    if cand_mae > inc_mae + tolerance:
        return False, (f"candidate MAE {cand_mae} worse than incumbent {inc_mae} "
                       f"+ tol {tolerance}")

    # Anti-ratchet: measured against the best ever, which never moves upward.
    if best_mae is not None and cand_mae > best_mae + max_drift:
        return False, (f"candidate MAE {cand_mae} has drifted more than "
                       f"{max_drift} from the best ever recorded ({best_mae}) — "
                       f"not promoting. The model is degrading night over night; "
                       f"investigate before this is relaxed.")

    reason = f"candidate MAE {cand_mae} <= incumbent {inc_mae} + tol {tolerance}"
    if best_mae is not None:
        drift = round(cand_mae - best_mae, 3)
        reason += f"; drift from best {best_mae} is {drift:+} (cap {max_drift})"
    return True, reason


def serving_config(split_date: str | None = None) -> EngineConfig:
    """THE config. Training, gating and serving must all use this one object.

    This exists because they did not. `price_and_report` shipped market_led=True
    while the gate, `glowstar.status` and every backtest built a bare
    EngineConfig() (market_led=False) — so the accuracy we measured and published
    was from a pipeline the client never received. On the same held-out stones the
    shipped path scored MAE 7.48 / bias +6.10 against the measured 3.84 / +0.85.
    A gate that scores a different config than it promotes is not a gate.

    Anything that prices a stone for real must construct its config HERE.
    """
    return EngineConfig(split_date=split_date or SETTINGS.backtest_split_date)


def _assert_gate_scores_what_ships(engine: PricingEngine) -> None:
    """Fail loudly if the gated engine is not on the serving config.

    A silent divergence here is what let an unmeasured pricing path reach the desk
    for weeks, so it is an assertion, not a log line.
    """
    ref = serving_config(engine.cfg.split_date)
    for field in ("market_led", "anchor_lambda", "apply_asking_offset", "use_trend"):
        got, want = getattr(engine.cfg, field), getattr(ref, field)
        if got != want:
            raise AssertionError(
                f"Promotion gate is scoring {field}={got!r} but serving uses {want!r}. "
                "The gate must score the config that ships (see serving_config)."
            )


def _evaluate(engine: PricingEngine, test) -> dict:
    """Score the engine out-of-time, with the grid joined POINT-IN-TIME.

    `predict()` deliberately attaches TODAY's grid when a caller has not supplied
    one — correct when serving, because the desk really does price from today's
    grid. It is wrong for a backtest: it shows a June sale the August grid, so
    the gate scored a feature distribution the model was never trained on
    (training joins per-row `OrderDate`). `predict()` documents that a
    backtesting caller must attach point-in-time beforehand — and this function,
    the one the promotion gate uses, did not.

    Measured on one model/test set: today's-grid 2.648 MAE vs point-in-time
    2.390. Every promote/reject decision so far was made on the wrong number.
    It happened to be pessimistic, but that direction was luck, not design —
    it is CLAUDE.md Trap 5 again: evaluate the pipeline that SHIPS.
    """
    _assert_gate_scores_what_ships(engine)
    if getattr(engine, "grid_history", None) is not None \
            and "grid_discount" not in test.columns:
        from ..market.grid_history import attach_grid
        test = attach_grid(test, engine.grid_history)   # asof=None => per-row OrderDate
    sugg = engine.predict(test)
    pred = np.array([s.suggested_discount for s in sugg])
    lo = np.array([s.ci_discount_low for s in sugg])
    hi = np.array([s.ci_discount_high for s in sugg])
    actual = test["FDiscount"].to_numpy()
    m = M.compute(pred, test)
    return {
        "mae": round(m.mae, 3),
        "within2": round(m.within2, 3),
        "within5": round(m.within5, 3),
        "coverage": round(M.interval_calibration(lo, hi, actual), 3),
        "bias": round(float(np.mean(pred - actual)), 3),
    }


def retrain(*, prefer_live: bool = True, split_date: str | None = None,
            tolerance: float = PROMOTE_TOLERANCE_PTS) -> dict:
    """Run one retrain cycle. Returns a summary dict (also logged)."""
    split_date = split_date or SETTINGS.backtest_split_date

    # 1. Rebuild records.json FRESH from the live API (current stock + latest sales
    #    + live BGM), unioned onto the deep-history base — so the model is never
    #    trained on a stale file (client rule: everything live).
    if prefer_live:
        try:
            from ..data.history import rebuild_records_from_live
            rebuild_records_from_live()
        except Exception:
            log.exception("Live rebuild failed; retraining on the existing records.json.")

    # 2. Train on the (freshly rebuilt) full sold history.
    from ..data.loaders import load_records, sold_stones
    sold = sold_stones(load_records()[0], drop_outliers=True)

    # Human feedback is OFF by default and must stay off until it is calibrated.
    # Measured (23k sold, 122 desk records): enabling it costs +0.93 MAE
    # (3.92 -> 4.85) -- online per-segment corrections +0.61, feedback training
    # labels +0.32. The desk's returned 'glow price' is an ASKING QUOTE, not a
    # realized sale, so training a sale-price model on it teaches the wrong
    # target; and build_corrections(min_support=3) shifts a whole price cell off
    # 3 stones. Left on, the promotion gate rejects every candidate and the
    # nightly retrain silently freezes. Re-enable with GS_USE_FEEDBACK=1 only
    # after raising min_support (~8-10), shrinking the offsets, and scoring BOTH
    # realized-sale MAE and variance-vs-desk-quote.
    use_fb = os.environ.get("GS_USE_FEEDBACK", "0") != "0"
    feedback = fbstore.load_all() if use_fb else []
    if not use_fb:
        log.info("Feedback DISABLED for training (GS_USE_FEEDBACK=0); "
                 "%d records on disk are recorded but not learned.",
                 len(fbstore.load_all()))

    # 3. Candidate evaluated out-of-time (honest, leakage-free).
    # PRODUCTION HORIZON, not a fixed twelve-week split — see GATE_HORIZON_DAYS.
    train, test, origin = gate_split(sold)
    log.info("gate protocol: %s | train %d (< %s) | test %d (next %dd)",
             METRIC_PROTOCOL, len(train), origin.date(), len(test), GATE_HORIZON_DAYS)
    cand_eval = PricingEngine(serving_config(split_date)).fit(
        train, feedback_records=feedback)
    metrics = _evaluate(cand_eval, test) if len(test) else {
        "mae": None, "within5": None, "coverage": None, "bias": None}

    # 4. Promotion gate vs the incumbent.
    _, incumbent = registry.load_current()
    # The INCUMBENT's number is only comparable if it answers the same question.
    # After a protocol change its card holds a figure measured a different way;
    # using it would repeat exactly the mistake this change exists to fix. Treat
    # a mismatched incumbent as "no incumbent" — the first model under a new
    # protocol establishes the baseline, and the ceiling still backstops it.
    inc_mae = None
    if incumbent:
        if incumbent.get("metric_protocol") == METRIC_PROTOCOL:
            inc_mae = incumbent.get("test_mae")
        else:
            log.warning(
                "incumbent %s was scored under protocol %r, not %r — its MAE is "
                "not comparable, so this candidate sets a fresh baseline.",
                incumbent.get("version"), incumbent.get("metric_protocol"),
                METRIC_PROTOCOL)
    # The best MAE ever recorded — a reference point that cannot drift upward.
    # Without it the gate is a ratchet; see gate_decision.
    best = registry.best_mae(protocol=METRIC_PROTOCOL)
    promote, reason = gate_decision(metrics["mae"], inc_mae, best, tolerance)
    if best is not None and metrics["mae"] is not None:
        log.info("gate: candidate %.3f | incumbent %s | best ever %.3f | drift %+.3f",
                 metrics["mae"], inc_mae, best, metrics["mae"] - best)

    # 5. Save a PRODUCTION engine (trained on ALL sold + feedback); promote if gated.
    prod = PricingEngine(serving_config(split_date)).fit(
        sold, feedback_records=feedback)
    version = datetime.now().strftime("%Y%m%dT%H%M%S")
    card = registry.ModelCard(
        version=version, trained_at=datetime.now().isoformat(timespec="seconds"),
        n_train=len(sold), test_mae=metrics["mae"],
        test_within2=metrics.get("within2"), test_within5=metrics["within5"],
        test_coverage=metrics["coverage"], promoted=promote, notes=reason,
        metric_protocol=METRIC_PROTOCOL)
    registry.save_engine(prod, card)
    if promote:
        registry.set_current(version)

    # Is the desk's feedback ready to TRAIN on yet? Reported every night so
    # "switch it on once there's enough data" is a measured event, not something
    # someone has to remember to check. It never flips the switch itself.
    try:
        from ..feedback.readiness import assess, format_report
        log.info("%s", format_report(assess()))
    except Exception:
        log.exception("Feedback readiness check failed (non-fatal).")

    # Housekeeping: a nightly ~9 MB artifact is ~3.2 GB/year and fills the disk of
    # a server nobody is watching. Keep the rollback window + the promoted history.
    try:
        pruned = registry.prune()
        if pruned["removed"]:
            log.info("Pruned %d old model artifacts (freed %.0f MB).",
                     len(pruned["removed"]), pruned["freed_mb"])
    except Exception:
        log.exception("Registry prune failed (non-fatal; the new model is saved).")

    # `len(test)`, not `info.n_test`. `info` never existed — it was left behind
    # when the gate moved to the rolling production horizon and `gate_split` began
    # returning (train, test, origin). It raised NameError AFTER the model was
    # saved and promoted, so every symptom pointed the wrong way: the log said
    # "Saved model ... promoted=True" and the model really was in the registry,
    # while the unit exited non-zero and systemd silently skipped everything
    # after it — the drift report, the nightly BACKUP, and the ExecStartPost that
    # restarts the API onto the model it had just promoted.
    summary = {"version": version, "promoted": promote, "reason": reason,
               "metrics": metrics, "n_sold": len(sold), "n_test": len(test),
               "n_feedback": len(feedback)}
    log.info("Retrain %s: promoted=%s (%s); metrics=%s", version, promote, reason, metrics)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r = retrain(prefer_live=True)
    print("=" * 70)
    print("RETRAIN CYCLE")
    print("=" * 70)
    print(f"version   : {r['version']}")
    print(f"sold rows : {r['n_sold']:,}   feedback: {r['n_feedback']}   test: {r['n_test']:,}")
    print(f"metrics   : {r['metrics']}")
    print(f"promoted  : {r['promoted']}  ({r['reason']})")
    print("=" * 70)


if __name__ == "__main__":
    main()
