"""Versioned model registry: persist trained engines and track the live one.

Today the engine is retrained in memory on every process start. For production
that is both slow and unaccountable (no record of WHICH model priced a stone).
This registry fixes both:

  * Each model is saved immutably under artifacts/models/<version>/ with the
    pickled engine and a metrics.json card (out-of-time accuracy, training rows,
    dates) so any past suggestion can be reconstructed (brief Section 2.5 audit).
  * A `current.json` pointer names the live version, so serving loads a ready
    model instead of retraining.
  * Nothing here auto-promotes — the retrain job (glowstar/training/retrain.py)
    gates on accuracy and only then calls set_current().
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import joblib
except ImportError:  # joblib ships with scikit-learn; fall back to pickle
    joblib = None
    import pickle

from ..config import ARTIFACTS_DIR

log = logging.getLogger(__name__)

MODELS_DIR = ARTIFACTS_DIR / "models"


@dataclass
class ModelCard:
    """Audit card stored next to every saved model."""

    version: str
    trained_at: str
    n_train: int
    test_mae: float | None = None
    test_within5: float | None = None
    test_coverage: float | None = None
    promoted: bool = False
    notes: str = ""


def _dump(obj, path: Path) -> None:
    if joblib is not None:
        joblib.dump(obj, path)
    else:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


def _load(path: Path):
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_engine(engine, card: ModelCard) -> Path:
    """Persist an engine + its card immutably under its version dir."""
    d = MODELS_DIR / card.version
    d.mkdir(parents=True, exist_ok=True)
    _dump(engine, d / "engine.joblib")
    (d / "metrics.json").write_text(json.dumps(asdict(card), indent=2), encoding="utf-8")
    log.info("Saved model %s (test MAE=%s, promoted=%s)",
             card.version, card.test_mae, card.promoted)
    return d


def set_current(version: str) -> None:
    """Point serving at `version` (called only after the promotion gate passes)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "current.json").write_text(
        json.dumps({"version": version}), encoding="utf-8")
    log.info("Promoted model %s to current.", version)


def current_version() -> str | None:
    p = MODELS_DIR / "current.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("version")


def load_current() -> tuple[object | None, dict | None]:
    """Return (engine, card_dict) for the live model, or (None, None)."""
    v = current_version()
    if v is None:
        return None, None
    d = MODELS_DIR / v
    if not (d / "engine.joblib").exists():
        log.warning("current.json points at %s but no engine file exists.", v)
        return None, None
    engine = _load(d / "engine.joblib")
    card = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    return engine, card


def list_versions() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())
