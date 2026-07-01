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

import logging

import pandas as pd

from ..ingestion.snapshots import SNAPSHOT_ROOT
from .loaders import load_records, sold_stones

log = logging.getLogger(__name__)


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
