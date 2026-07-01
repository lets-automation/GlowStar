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
from datetime import datetime

import numpy as np

from ..config import SETTINGS
from ..data.history import assemble_sold_history
from ..feedback import store as fbstore
from ..models.engine import PricingEngine, EngineConfig
from ..models import registry
from ..validation.backtest import time_split
from ..validation import metrics as M

log = logging.getLogger(__name__)

# A candidate may be at most this many MAE points worse than the incumbent and
# still be promoted (small wiggle for run-to-run noise). Configurable.
PROMOTE_TOLERANCE_PTS = 0.25


def gate_decision(cand_mae: float | None, inc_mae: float | None,
                  tolerance: float = PROMOTE_TOLERANCE_PTS) -> tuple[bool, str]:
    """Pure promotion rule (unit-testable): promote unless the candidate is
    materially worse than the incumbent on the out-of-time test window."""
    if cand_mae is None:
        return False, "no test window to evaluate — not promoting"
    if inc_mae is None:
        return True, "no incumbent — promoting first model"
    if cand_mae <= inc_mae + tolerance:
        return True, f"candidate MAE {cand_mae} <= incumbent {inc_mae} + tol {tolerance}"
    return False, f"candidate MAE {cand_mae} worse than incumbent {inc_mae} + tol {tolerance}"


def _evaluate(engine: PricingEngine, test) -> dict:
    sugg = engine.predict(test)
    pred = np.array([s.suggested_discount for s in sugg])
    lo = np.array([s.ci_discount_low for s in sugg])
    hi = np.array([s.ci_discount_high for s in sugg])
    actual = test["FDiscount"].to_numpy()
    m = M.compute(pred, test)
    return {
        "mae": round(m.mae, 3),
        "within5": round(m.within5, 3),
        "coverage": round(M.interval_calibration(lo, hi, actual), 3),
        "bias": round(float(np.mean(pred - actual)), 3),
    }


def retrain(*, prefer_live: bool = True, split_date: str | None = None,
            tolerance: float = PROMOTE_TOLERANCE_PTS) -> dict:
    """Run one retrain cycle. Returns a summary dict (also logged)."""
    split_date = split_date or SETTINGS.backtest_split_date

    # 1. Bank a fresh snapshot if we can (side effect: grows the history).
    if prefer_live:
        try:
            from ..pipeline import ingest_records
            ingest_records(prefer_live=True)
        except Exception:
            log.exception("Live pull failed; retraining on existing snapshots/file.")

    # 2. Assemble the union of sold history across all banked snapshots.
    sold = assemble_sold_history()
    feedback = fbstore.load_all()

    # 3. Candidate evaluated out-of-time (honest, leakage-free).
    train, test, info = time_split(sold, split_date)
    cand_eval = PricingEngine(EngineConfig(split_date=split_date)).fit(
        train, feedback_records=feedback)
    metrics = _evaluate(cand_eval, test) if len(test) else {
        "mae": None, "within5": None, "coverage": None, "bias": None}

    # 4. Promotion gate vs the incumbent.
    _, incumbent = registry.load_current()
    inc_mae = (incumbent or {}).get("test_mae")
    promote, reason = gate_decision(metrics["mae"], inc_mae, tolerance)

    # 5. Save a PRODUCTION engine (trained on ALL sold + feedback); promote if gated.
    prod = PricingEngine(EngineConfig(split_date=split_date)).fit(
        sold, feedback_records=feedback)
    version = datetime.now().strftime("%Y%m%dT%H%M%S")
    card = registry.ModelCard(
        version=version, trained_at=datetime.now().isoformat(timespec="seconds"),
        n_train=len(sold), test_mae=metrics["mae"], test_within5=metrics["within5"],
        test_coverage=metrics["coverage"], promoted=promote, notes=reason)
    registry.save_engine(prod, card)
    if promote:
        registry.set_current(version)

    summary = {"version": version, "promoted": promote, "reason": reason,
               "metrics": metrics, "n_sold": len(sold), "n_test": info.n_test,
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
