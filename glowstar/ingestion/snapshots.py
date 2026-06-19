"""Immutable, timestamped snapshot store with schema-drift detection (3.5).

Every pull is persisted under data/snapshots/<source>/<YYYY-MM-DD>.json and is
never overwritten — the whole point is to grow the time series the inventory and
market-trend models need. Loads are idempotent (keyed on source + date); a
second pull on the same day verifies rather than clobbers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..config import DATA_DIR

log = logging.getLogger(__name__)

SNAPSHOT_ROOT = DATA_DIR / "snapshots"


@dataclass
class SnapshotResult:
    path: Path
    source: str
    snapshot_date: str
    n_records: int
    sha256: str
    already_existed: bool
    added_fields: list[str]
    removed_fields: list[str]


def _digest(records: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _latest_prior(source_dir: Path, before: str) -> list[dict] | None:
    if not source_dir.exists():
        return None
    priors = sorted(p for p in source_dir.glob("*.json") if p.stem < before)
    if not priors:
        return None
    return json.loads(priors[-1].read_text(encoding="utf-8"))


def _field_set(records: list[dict]) -> set[str]:
    fields: set[str] = set()
    for r in records[:500]:                      # sample is enough for drift
        fields |= set(r.keys())
    return fields


def save_snapshot(records: list[dict], source: str,
                  snapshot_date: str | None = None) -> SnapshotResult:
    """Persist a snapshot immutably; detect schema drift vs the prior snapshot."""
    snapshot_date = snapshot_date or date.today().isoformat()
    source_dir = SNAPSHOT_ROOT / source
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{snapshot_date}.json"
    new_digest = _digest(records)

    prior = _latest_prior(source_dir, snapshot_date)
    prior_fields = _field_set(prior) if prior else set()
    new_fields = _field_set(records)
    added = sorted(new_fields - prior_fields)
    removed = sorted(prior_fields - new_fields)
    if added or removed:
        log.warning("Schema drift in %s vs prior: +%s -%s", source, added, removed)

    if path.exists():
        existing_digest = _digest(json.loads(path.read_text(encoding="utf-8")))
        if existing_digest != new_digest:
            log.warning("Snapshot %s/%s already exists with different content; "
                        "keeping the original (immutable).", source, snapshot_date)
        return SnapshotResult(path, source, snapshot_date, len(records),
                              existing_digest, True, added, removed)

    path.write_text(json.dumps(records, default=str), encoding="utf-8")
    log.info("Saved snapshot %s/%s (%s records)", source, snapshot_date, f"{len(records):,}")
    return SnapshotResult(path, source, snapshot_date, len(records),
                          new_digest, False, added, removed)
