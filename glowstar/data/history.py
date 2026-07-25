"""Assemble the full sold-stone training history across banked snapshots.

Why this is needed (verified): a single live `GetAllRecord` pull is a *current*
snapshot, and it returned FEWER sold stones than an earlier extract (live 19,579
vs shipped 20,143) — the API serves a rolling window, so each pull silently drops
the oldest sales. Training on only the latest pull would therefore shrink the
history over time. Unioning every immutable banked snapshot (deduped by StoneId,
keeping the most recent version of each stone) is what actually GROWS the
trainable history — the whole point of the daily snapshot job (brief Section 3.5).
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

from ..config import PATHS, REPO_ROOT
from ..ingestion.snapshots import SNAPSHOT_ROOT
from .loaders import load_records, sold_stones

log = logging.getLogger(__name__)

# Deep-history base kept once (the shipped extract that predates the API's rolling
# window). The live pull is UNIONed onto it so December history is never lost.
_HISTORY_BASE = REPO_ROOT / "records_pre_bgm.json"


def rebuild_records_from_live() -> int:
    """Re-pull the LIVE inventory+sales and UNION it onto the deep-history base,
    then overwrite records.json — so the training file is always FRESH (current
    stock, latest sales, live BGM) yet keeps the full history the rolling API
    window drops. Returns the record count. Best-effort: on a live failure it
    leaves the existing records.json untouched.

    This is what makes the nightly retrain 'all live' (client rule): the model is
    always trained on a freshly-rebuilt file, never a stale committed snapshot.
    """
    from ..ingestion import channel_partner
    fresh = channel_partner.get_all_records()

    # Union, by StoneId, in ascending freshness so the freshest wins:
    #   December deep-history base  ->  every banked daily snapshot  ->  the LIVE pull.
    # The accruing snapshots fill any gap between the fixed base and the API's
    # rolling window (as the window rolls forward the base alone would leave a hole);
    # the live pull always wins (current status + live BGM). This is the whole point
    # of banking daily snapshots — never lose history.
    by_id: dict = {}
    base_found = _HISTORY_BASE.exists()
    if base_found:
        for r in json.loads(_HISTORY_BASE.read_text(encoding="utf-8")):
            if r.get("StoneId"):
                by_id[r["StoneId"]] = r
    base_ids = set(by_id)
    snap_dir = SNAPSHOT_ROOT / "channel_partner"
    n_snaps = 0
    if snap_dir.exists():
        for snap in sorted(snap_dir.glob("*.json")):          # ascending by date
            try:
                for r in json.loads(snap.read_text(encoding="utf-8")):
                    if r.get("StoneId"):
                        by_id[r["StoneId"]] = r
                n_snaps += 1
            except (ValueError, OSError):
                continue
    fresh_ids = {r.get("StoneId") for r in fresh}
    for r in fresh:                                            # LIVE wins ties
        if r.get("StoneId"):
            by_id[r["StoneId"]] = r

    combined = list(by_id.values())
    # Atomic write (temp + os.replace) so a crash mid-write can never leave a
    # truncated records.json that a concurrent read/train would choke on.
    tmp = PATHS.records.with_suffix(".tmp")
    tmp.write_text(json.dumps(combined), encoding="utf-8")
    os.replace(tmp, PATHS.records)
    # A fresh deploy where NEITHER the deep-history base NOR any snapshot exists
    # means the model can only ever see the API's rolling window (oldest sales
    # silently dropped). Make that LOUD rather than silent.
    if not base_found and n_snaps == 0:
        log.warning("rebuild_records_from_live: NO deep-history base (%s) and NO banked "
                    "snapshots — training history is limited to the live rolling window; "
                    "the oldest sales will be lost. Provide records_pre_bgm.json out-of-band.",
                    _HISTORY_BASE.name)
    base_only = len(base_ids - fresh_ids)                     # history the live pull lacks
    log.info("Rebuilt records.json: %d records (%d live + %d snapshots + base[found=%s, "
             "history-only rows=%d]).", len(combined), len(fresh), n_snaps, base_found, base_only)
    return len(combined)


def assemble_sold_history(source: str = "channel_partner", *,
                          drop_outliers: bool = True) -> pd.DataFrame:
    """Union sold stones across all banked snapshots; dedupe by StoneId.

    Falls back to the shipped records.json when no snapshots have been banked yet,
    so this works from day one and strengthens automatically as snapshots accrue.
    """
    snap_dir = SNAPSHOT_ROOT / source
    snaps = sorted(snap_dir.glob("*.json")) if snap_dir.exists() else []

    if not snaps:
        df, _ = load_records()
        sold = sold_stones(df, drop_outliers=drop_outliers)
        log.info("No banked snapshots; using shipped records.json (%s sold).", f"{len(sold):,}")
        return sold

    frames = []
    for snap in snaps:                                    # ascending by date
        df, _ = load_records(snap)
        frames.append(sold_stones(df, drop_outliers=drop_outliers))
    allsold = pd.concat(frames, ignore_index=True)
    before = len(allsold)
    # keep="last" -> the most recent snapshot's version of a stone wins.
    allsold = allsold.drop_duplicates(subset="StoneId", keep="last").reset_index(drop=True)
    log.info("Assembled sold history from %d snapshots: %s unique sold (%s rows before dedupe).",
             len(snaps), f"{len(allsold):,}", f"{before:,}")
    return allsold
