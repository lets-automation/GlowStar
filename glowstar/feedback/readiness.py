"""Is the desk's feedback ready to TRAIN on yet? — answered automatically.

THE PROBLEM THIS SOLVES
-----------------------
Feedback is stored but deliberately not trained on. The desk's returned price is
usually an ASKING QUOTE, not the price a stone sold for, and training a
sale-price model on quote labels teaches the wrong target: measured, it cost
+0.93 MAE. So it is off.

"Switch it on once there is enough data" is a promise nobody can act on unless
someone remembers to check. This module removes the remembering: the nightly job
runs it, and it reports — with numbers — whether the evidence now supports
turning feedback on, and what is still missing if not.

It never flips the switch by itself. Enabling feedback changes every price the
desk sees, so it stays a human decision (and a §12 amendment under the MOU) —
but a decision made against evidence rather than a hunch.

THE TWO GATES
-------------
1. VOLUME  — enough overrides, spread across enough price cells, to estimate a
   correction that is not just three stones' noise.
2. PROOF   — a candidate trained WITH feedback must beat the current model on
   BOTH objectives at once:
       (a) realized-sale accuracy  — are we still predicting what stones sell for?
       (b) desk agreement          — are we closer to what the desk quotes?
   Improving (b) while wrecking (a) is exactly the failure mode; requiring both
   is what makes switching it on safe rather than hopeful.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from ..market.segments import segment_keys
from .store import Decision, load_all

log = logging.getLogger(__name__)

# Volume gate. `build_corrections(min_support=3)` moved a whole price cell off
# three stones — far too few. These are the levels at which a per-cell correction
# starts to be an estimate rather than an anecdote.
MIN_OVERRIDES = 150            # total priced corrections on record
MIN_CELLS = 25                 # distinct price cells with >= MIN_PER_CELL each
MIN_PER_CELL = 8               # per-cell support (the calibrated min_support)


def _cell(rec: dict) -> str:
    """The most specific segment key a feedback record belongs to."""
    try:
        key = segment_keys(rec.get("shape_full"), float(rec.get("weight") or 0),
                           rec.get("color"), rec.get("clarity"))[0]
        return "|".join(map(str, key))
    except Exception:
        return "?"


def assess(records: list[dict] | None = None) -> dict:
    """Report whether feedback is ready to train on, and what is missing if not."""
    recs = load_all() if records is None else records
    overrides = [r for r in recs
                 if r.get("decision") == Decision.OVERRIDE.value
                 and r.get("human_discount") is not None]

    cells = Counter(_cell(r) for r in overrides)
    strong = {c: n for c, n in cells.items() if n >= MIN_PER_CELL and c != "?"}

    deltas = np.array([float(r["human_discount"]) - float(r["suggested_discount"])
                       for r in overrides]) if overrides else np.array([])

    volume_ok = len(overrides) >= MIN_OVERRIDES and len(strong) >= MIN_CELLS
    missing: list[str] = []
    if len(overrides) < MIN_OVERRIDES:
        missing.append(f"{MIN_OVERRIDES - len(overrides)} more priced overrides "
                       f"({len(overrides)}/{MIN_OVERRIDES})")
    if len(strong) < MIN_CELLS:
        missing.append(f"{MIN_CELLS - len(strong)} more price cells with "
                       f"{MIN_PER_CELL}+ corrections ({len(strong)}/{MIN_CELLS})")

    out = {
        "ready_to_test": volume_ok,
        "n_decisions": len(recs),
        "n_overrides": len(overrides),
        "n_cells_with_support": len(strong),
        "still_needed": missing,
        "reason_coverage": round(
            sum(1 for r in overrides if r.get("reason_code")) / max(1, len(overrides)), 3),
    }
    if len(deltas):
        out["desk_moves_us"] = {
            "median_pts": round(float(np.median(deltas)), 2),
            "mean_pts": round(float(np.mean(deltas)), 2),
            "share_deeper": round(float(np.mean(deltas < 0)), 3),
        }
    out["next_step"] = (
        "Volume gate PASSED — run the A/B (see run_ab) and only enable "
        "GS_USE_FEEDBACK=1 if BOTH objectives improve."
        if volume_ok else
        "Keep collecting. Feedback stays off; the model keeps learning from real "
        "sales nightly, which is unaffected by this."
    )
    return out


def format_report(a: dict | None = None) -> str:
    """One short block for the nightly log — readable without opening a notebook."""
    a = a or assess()
    lines = [
        "feedback readiness:",
        f"  decisions {a['n_decisions']} | priced overrides {a['n_overrides']}"
        f"/{MIN_OVERRIDES} | supported cells {a['n_cells_with_support']}/{MIN_CELLS}",
    ]
    if a.get("desk_moves_us"):
        d = a["desk_moves_us"]
        lines.append(f"  desk moves us {d['median_pts']:+.2f} pts (median); "
                     f"{d['share_deeper']:.0%} of the time deeper")
    if a["still_needed"]:
        lines.append("  still needed: " + "; ".join(a["still_needed"]))
    lines.append(f"  -> {a['next_step']}")
    return "\n".join(lines)


def main() -> None:                       # python -m glowstar.feedback.readiness
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(format_report())


if __name__ == "__main__":
    main()
