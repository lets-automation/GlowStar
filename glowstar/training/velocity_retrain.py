"""Nightly velocity retrain with a promotion gate (MOU 5.1, 5.7).

Mirrors `training/retrain.py` deliberately, down to the anti-ratchet, because
that pattern has already caught a bad pricing model in production. The
Workstream-A gate is never bypassed and neither is this one.

WHAT THE GATE SCORES IS WHAT SHIPS
-----------------------------------
`_assert_gate_scores_what_ships()` fails the retrain if the candidate is not on
`velocity.serving_velocity_config()`. This is not defensive decoration: the
pricing engine shipped one configuration while its gate scored another for
weeks, the published accuracy was from a pipeline the client never received, and
on the same stones the shipped path was roughly twice the error. The same
mistake is available here every time a knob is added.

THE GATE USES TWO METRICS, NOT ONE
-----------------------------------
A velocity model can be made to look better on either axis alone:

  * **C-index** — does it RANK stones correctly? It is blind to any monotone
    distortion, so a model can be badly over-confident and score perfectly.
    Measured here: an off-by-one that reported P(sold by 30d) as if it were day
    45 moved the C-index by nothing at all.
  * **Calibration error** — are the PROBABILITIES right? It is blind to ranking.

Both are gated, because the desk reads both: the class comes from the ranking
and the days-to-sell number comes from the probabilities. The tuning sweep
showed this concretely — going from 60 to 300 boosting iterations left the
C-index flat and made calibration three times worse.

THE BASELINE IS THE LIVE FIELD, NOT A STRAW MAN
------------------------------------------------
The candidate must beat a segment-median baseline, which is essentially what
`service/tradeability.py` publishes on `/frontoffice/reason` today. Failing that
means the model is not worth binding the client's screen to.

Run:  python -m glowstar.training.velocity_retrain
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR, SETTINGS
from ..inventory.survival import build_survival_frame
from ..inventory.velocity import VelocityModel, serving_velocity_config
from ..models.registry import _dump, _load
from ..validation import survival_backtest as B

log = logging.getLogger(__name__)

VELOCITY_DIR = ARTIFACTS_DIR / "velocity_models"

# Promotion thresholds. Deliberately explicit rather than derived, and each one
# is a number a human can argue with.
MIN_C_INDEX = 0.55                 # below this it is barely ranking at all
C_INDEX_TOLERANCE = 0.01           # day-to-day noise the gate tolerates
MAX_C_DRIFT_FROM_BEST = 0.03       # anti-ratchet: the best ever only moves up
MAX_CALIBRATION_ERROR = 0.15       # weighted |predicted - observed| P(sold)
# Which question the metrics answer. Comparing across protocols is meaningless,
# and doing it silently is how the pricing gate nearly froze the model.
METRIC_PROTOCOL = "listing_split_admin_censored_v1"


@dataclass
class VelocityCard:
    """Audit card stored next to every saved velocity model."""

    version: str
    trained_at: str
    n_train: int
    n_train_events: int
    n_test: int
    n_test_events: int
    c_index: float | None = None
    c_index_baseline: float | None = None
    calibration_error: float | None = None
    comparable_pairs: int = 0
    split_date: str = ""
    dropped_features: tuple[str, ...] = ()
    promoted: bool = False
    notes: str = ""
    metric_protocol: str = METRIC_PROTOCOL


def _assert_gate_scores_what_ships(model: VelocityModel) -> None:
    """Fail loudly if the gated model is not on the serving config."""
    ref = serving_velocity_config()
    if model.cfg != ref:
        raise AssertionError(
            "The velocity promotion gate is scoring a configuration that is not "
            "the one that ships. Construct it from "
            "`inventory.velocity.serving_velocity_config()` or it does not ship."
        )


def calibration_error(result: dict) -> float | None:
    """One number for "are the probabilities right?", weighted by bin size.

    Averaged over the horizons that actually have follow-up. A horizon nothing
    has been followed to contributes nothing rather than a zero — scoring an
    unobservable horizon as perfect is how a gate stops gating.
    """
    errs, weights = [], []
    for _h, rows in (result.get("calibration") or {}).items():
        usable = [r for r in rows if r.get("observed") is not None]
        if not usable:
            continue
        n = sum(r["n"] for r in usable)
        if not n:
            continue
        errs.append(sum(abs(r["gap"]) * r["n"] for r in usable) / n)
        weights.append(n)
    if not errs:
        return None
    return float(np.average(errs, weights=weights))


def gate_decision(cand_c: float | None, cand_cal: float | None,
                  baseline_c: float | None, inc_c: float | None,
                  best_c: float | None = None,
                  *, min_c: float = MIN_C_INDEX,
                  tolerance: float = C_INDEX_TOLERANCE,
                  max_drift: float = MAX_C_DRIFT_FROM_BEST,
                  max_cal: float = MAX_CALIBRATION_ERROR) -> tuple[bool, str]:
    """Pure promotion rule (unit-testable). ALL of these must hold:

      1. the candidate ranks meaningfully better than a coin (`min_c`);
      2. it beats the SEGMENT-MEDIAN baseline — the field that ships today;
      3. its probabilities are calibrated within `max_cal`;
      4. it is not materially worse than the INCUMBENT (day-to-day); and
      5. it has not drifted too far from the BEST ever seen (anti-ratchet).

    Rule 5 exists because gating on the incumbent alone is a ratchet: each night
    the bar moves to wherever the model landed, so it can degrade without bound
    while every single night passes.
    """
    if cand_c is None:
        return False, "no test window to evaluate — not promoting"
    if cand_c < min_c:
        return False, (f"candidate C-index {cand_c} is below the floor {min_c} — "
                       f"it is barely ranking stones at all")
    if baseline_c is not None and cand_c <= baseline_c:
        return False, (f"candidate C-index {cand_c} does not beat the "
                       f"segment-median baseline {baseline_c}, which is what "
                       f"tradeability.py already publishes — nothing to gain")
    if cand_cal is not None and cand_cal > max_cal:
        return False, (f"candidate calibration error {cand_cal:.3f} exceeds "
                       f"{max_cal} — it ranks acceptably but its days-to-sell "
                       f"probabilities are wrong, and the desk reads those")
    if inc_c is None:
        return True, "no incumbent — promoting first velocity model"
    if cand_c < inc_c - tolerance:
        return False, (f"candidate C-index {cand_c} worse than incumbent "
                       f"{inc_c} - tol {tolerance}")
    if best_c is not None and cand_c < best_c - max_drift:
        return False, (f"candidate C-index {cand_c} has drifted more than "
                       f"{max_drift} below the best ever recorded ({best_c}) — "
                       f"not promoting. Investigate before this is relaxed.")
    reason = f"C-index {cand_c} >= incumbent {inc_c} - tol {tolerance}"
    if baseline_c is not None:
        reason += f"; beats baseline {baseline_c}"
    if cand_cal is not None:
        reason += f"; calibration {cand_cal:.3f} <= {max_cal}"
    if best_c is not None:
        reason += f"; drift from best {best_c} is {cand_c - best_c:+.3f} (cap {max_drift})"
    return True, reason


# ---------------------------------------------------------------------------
# registry (same shape as models/registry.py, kept separate so a velocity model
# can never be loaded where a pricing engine is expected)
# ---------------------------------------------------------------------------
def save(model: VelocityModel, card: VelocityCard) -> Path:
    d = VELOCITY_DIR / card.version
    d.mkdir(parents=True, exist_ok=True)
    _dump(model, d / "model.joblib")
    (d / "metrics.json").write_text(json.dumps(asdict(card), indent=2, default=str),
                                    encoding="utf-8")
    log.info("Saved velocity model %s (C=%s, calib=%s, promoted=%s)",
             card.version, card.c_index, card.calibration_error, card.promoted)
    return d


def set_current(version: str) -> None:
    VELOCITY_DIR.mkdir(parents=True, exist_ok=True)
    (VELOCITY_DIR / "current.json").write_text(json.dumps({"version": version}),
                                               encoding="utf-8")
    log.info("Promoted velocity model %s to current.", version)


def current_version() -> str | None:
    p = VELOCITY_DIR / "current.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("version")


def load_current() -> tuple[VelocityModel | None, dict | None]:
    v = current_version()
    if v is None:
        return None, None
    d = VELOCITY_DIR / v
    if not (d / "model.joblib").exists():
        log.warning("velocity current.json points at %s but no model file exists.", v)
        return None, None
    return _load(d / "model.joblib"), json.loads(
        (d / "metrics.json").read_text(encoding="utf-8"))


def _cards() -> list[dict]:
    if not VELOCITY_DIR.exists():
        return []
    out = []
    for d in sorted(VELOCITY_DIR.iterdir()):
        f = d / "metrics.json"
        if d.is_dir() and f.exists():
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return out


def best_c_index(protocol: str = METRIC_PROTOCOL) -> float | None:
    """Highest C-index ever recorded, ONLY among cards on the same protocol.

    Cards scored under a different protocol are excluded rather than silently
    mixed in — the pricing registry learned that the expensive way.
    """
    best = None
    for card in _cards():
        if card.get("metric_protocol") != protocol:
            continue
        c = card.get("c_index")
        if isinstance(c, (int, float)) and (best is None or c > best):
            best = float(c)
    return best


def incumbent_c_index() -> float | None:
    _, card = load_current()
    if not card or card.get("metric_protocol") != METRIC_PROTOCOL:
        return None
    c = card.get("c_index")
    return float(c) if isinstance(c, (int, float)) else None


# ---------------------------------------------------------------------------
def retrain(split_date: str | None = None, *, promote: bool = True,
            frame: pd.DataFrame | None = None) -> dict:
    """Fit a candidate, score it out-of-time, gate it, and only then promote."""
    split_date = split_date or SETTINGS.backtest_split_date
    if frame is None:
        frame, _ = build_survival_frame()

    result = B.evaluate(frame, split_date, cfg=serving_velocity_config())
    cal = calibration_error(result)

    # The model that actually SHIPS is fitted on everything, not on the training
    # half — the gate's job is to prove the recipe generalises, and then the
    # recipe is applied to all the data we have.
    shipped = VelocityModel(serving_velocity_config()).fit(frame)
    _assert_gate_scores_what_ships(shipped)

    version = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
    inc, best = incumbent_c_index(), best_c_index()
    ok, reason = gate_decision(result["c_index_model"], cal,
                               result["c_index_segment_median"], inc, best)

    card = VelocityCard(
        version=version, trained_at=pd.Timestamp.now().isoformat(timespec="seconds"),
        n_train=result["n_train"], n_train_events=result["n_train_events"],
        n_test=result["n_test"], n_test_events=result["n_test_events"],
        c_index=result["c_index_model"],
        c_index_baseline=result["c_index_segment_median"],
        calibration_error=None if cal is None else round(cal, 4),
        comparable_pairs=result["comparable_pairs"], split_date=split_date,
        dropped_features=tuple(shipped.dropped_features_),
        promoted=bool(ok and promote), notes=reason,
    )
    save(shipped, card)
    if ok and promote:
        set_current(version)
    else:
        log.warning("Velocity model %s NOT promoted: %s", version, reason)
    return {"version": version, "promoted": card.promoted, "reason": reason,
            "c_index": card.c_index, "c_index_baseline": card.c_index_baseline,
            "calibration_error": card.calibration_error,
            "incumbent_c_index": inc, "best_c_index": best}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = retrain()
    print("\n" + "=" * 66)
    print("VELOCITY RETRAIN")
    print("=" * 66)
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
