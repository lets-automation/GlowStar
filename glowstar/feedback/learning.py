"""Learn from human feedback (client request: "the model learns from it too").

Three mechanisms, from fastest to most durable:

  1. ONLINE CORRECTION (immediate). From recent OVERRIDE decisions we compute a
     per-segment offset = median(human_discount - suggested_discount), with
     hierarchical backoff. The engine applies it to future suggestions in that
     segment right away — so repeated rejections shift pricing without waiting
     for a retrain.

  2. RETRAIN LABELS (durable). Every OVERRIDE is a gold label (the human's
     correct discount for a real stone) and every ACCEPT confirms the suggestion;
     both are appended as weighted training rows on the next fit, so the model
     converges to human-validated pricing.

  3. REASON ANALYTICS (directing). Aggregated reason codes tell us WHAT to fix
     (e.g. many BGM_PRESENT -> prioritise CRM BGM capture; many MARKET_MOVED ->
     refresh the anchor / increase trend damping).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ..market.segments import segment_keys
from .store import Decision

# Columns a feedback record carries that map to model features.
_FEATURE_COLS = {
    "shape_full": "Shape_full", "weight": "Weight", "color": "Color",
    "clarity": "Clarity", "cps": "CPS", "fluorescence": "Fluorescence",
    "lab": "Lab", "location": "Location", "rap": "Rap",
}


def build_corrections(records: list[dict], min_support: int = 3) -> dict[str, dict]:
    """Per-segment correction offsets from OVERRIDE decisions (hierarchical)."""
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        if r.get("decision") != Decision.OVERRIDE.value or r.get("human_discount") is None:
            continue
        delta = float(r["human_discount"]) - float(r["suggested_discount"])
        for key in segment_keys(r["shape_full"], r["weight"], r["color"], r["clarity"]):
            buckets[key].append(delta)
    table: dict[str, dict] = {}
    for key, deltas in buckets.items():
        if len(deltas) >= min_support:
            name = "|".join(map(str, key)) if key else "__global__"
            table[name] = {"offset": round(float(np.median(deltas)), 2), "n": len(deltas)}
    return table


def correction_for(table: dict[str, dict], shape: str, weight: float,
                   color: str, clarity: str) -> float:
    """Most specific available segment correction for a stone (else 0)."""
    for key in segment_keys(shape, weight, color, clarity):
        name = "|".join(map(str, key)) if key else "__global__"
        if name in table:
            return table[name]["offset"]
    return 0.0


def as_training_examples(records: list[dict], accept_weight: float = 1.0,
                         override_weight: float = 3.0):
    """Turn decisions into (X-ready rows, labels, weights) for the next retrain.

    OVERRIDE -> label = human_discount (gold, up-weighted).
    ACCEPT   -> label = suggested_discount (confirmation).
    REJECT without a price gives no usable label and is skipped (its reason still
    feeds analytics).
    """
    rows, labels, weights = [], [], []
    for r in records:
        d = r.get("decision")
        if d == Decision.OVERRIDE.value and r.get("human_discount") is not None:
            label, w = float(r["human_discount"]), override_weight
        elif d == Decision.ACCEPT.value:
            label, w = float(r["suggested_discount"]), accept_weight
        else:
            continue
        row = {model_col: r.get(fb_key) for fb_key, model_col in _FEATURE_COLS.items()}
        # Feedback has no transaction dates; stamp with the decision date so the
        # recency weighting treats human labels as current.
        row["OrderDate_dt"] = pd.to_datetime(r.get("timestamp"), utc=True, errors="coerce")
        if pd.notna(row["OrderDate_dt"]):
            row["OrderDate_dt"] = row["OrderDate_dt"].tz_localize(None)
        row["MarketSheetDate_dt"] = row["OrderDate_dt"]
        row["FDiscount"] = label
        rows.append(row)
        labels.append(label)
        weights.append(w)
    if not rows:
        return pd.DataFrame(), np.array([]), np.array([])
    return pd.DataFrame(rows), np.array(labels), np.array(weights)


def reason_summary(records: list[dict]) -> dict:
    """Counts of rejection reasons + acceptance rate — directs what to fix next."""
    decisions = Counter(r.get("decision") for r in records)
    reasons = Counter(r.get("reason_code") for r in records if r.get("reason_code"))
    n = len(records) or 1
    return {
        "n_decisions": len(records),
        "acceptance_rate": round(decisions.get(Decision.ACCEPT.value, 0) / n, 3),
        "decisions": dict(decisions),
        "reasons": dict(reasons.most_common()),
    }
