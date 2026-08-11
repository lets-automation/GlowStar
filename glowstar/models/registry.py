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
    test_within2: float | None = None
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


def best_mae() -> float | None:
    """Lowest test MAE ever recorded across every saved model card.

    The promotion gate needs a reference point that does NOT move upward. The
    incumbent's MAE is not that: it is replaced by every promotion, so gating on
    it alone is a ratchet that lets the model degrade without bound (see
    `gate_decision`). The best ever achieved only ever moves down.

    Reads the cards rather than the engines — cheap, and it still works for
    versions whose engine file has since been pruned off disk.
    """
    best: float | None = None
    if not MODELS_DIR.exists():
        return None
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        f = d / "metrics.json"
        if not f.exists():
            continue
        try:
            mae = json.loads(f.read_text(encoding="utf-8")).get("test_mae")
        except Exception:      # a corrupt card must not break the retrain
            continue
        if isinstance(mae, (int, float)) and (best is None or mae < best):
            best = float(mae)
    return best


def list_versions() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())


# A nightly retrain writes an ~9 MB engine every day: ~3.2 GB a year, and the disk
# fills silently on a server nobody is watching. Keep enough to audit and roll
# back, delete the rest.
KEEP_RECENT = 14          # ~2 weeks of daily retrains — the practical rollback window
KEEP_PROMOTED = 5         # plus the last few that actually SERVED prices


def prune(keep_recent: int = KEEP_RECENT, keep_promoted: int = KEEP_PROMOTED,
          dry_run: bool = False) -> dict:
    """Delete old model artifacts, keeping what audit and rollback actually need.

    ALWAYS kept, regardless of the limits:
      * the version `current.json` points at — deleting the live model would take
        serving down at the next restart;
      * the most recent `keep_recent` versions (rollback window);
      * the most recent `keep_promoted` versions that were PROMOTED — a promoted
        model priced real stones, so its card is the audit trail for those quotes.

    The metrics.json card is kept for every version even when its engine file is
    deleted: the card is tiny and is what proves what accuracy was claimed when.
    """
    versions = list_versions()
    cur = current_version()
    keep: set[str] = set(versions[-keep_recent:]) | ({cur} if cur else set())

    promoted: list[str] = []
    for v in versions:
        card_p = MODELS_DIR / v / "metrics.json"
        try:
            if json.loads(card_p.read_text(encoding="utf-8")).get("promoted"):
                promoted.append(v)
        except (OSError, ValueError):
            keep.add(v)                   # unreadable card: never silently delete
    keep |= set(promoted[-keep_promoted:])

    freed, removed = 0, []
    for v in versions:
        if v in keep:
            continue
        eng = MODELS_DIR / v / "engine.joblib"
        if not eng.exists():
            continue
        size = eng.stat().st_size
        if not dry_run:
            eng.unlink()                  # card stays: the audit trail survives
        freed += size
        removed.append(v)
    log.info("Registry prune: removed %d engine files, freed %.0f MB (kept %d)%s",
             len(removed), freed / 1e6, len(keep), " [dry run]" if dry_run else "")
    return {"removed": removed, "freed_mb": round(freed / 1e6, 1),
            "kept": sorted(keep), "dry_run": dry_run}
