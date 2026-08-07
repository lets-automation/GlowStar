"""Immutable feedback store: every human decision on a suggestion, with reason.

When a pricer accepts, rejects, or overrides a suggestion we record it append-
only (an audit trail and a training signal). A rejection MUST carry a reason
code — that is the whole point: the model learns not just *that* it was wrong but
*why*, and an override carries the human's corrected discount as a gold label.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from ..config import DATA_DIR

log = logging.getLogger(__name__)

FEEDBACK_LOG = DATA_DIR / "feedback" / "decisions.jsonl"


class Decision(str, Enum):
    ACCEPT = "accept"        # used the suggestion as-is
    OVERRIDE = "override"    # rejected; priced it differently (gold label)
    REJECT = "reject"        # rejected; no replacement given


class ReasonCode(str, Enum):
    """Why a suggestion was rejected — drives both learning and analytics."""

    DISCOUNT_TOO_DEEP = "discount_too_deep"     # price too low
    DISCOUNT_TOO_SHALLOW = "discount_too_shallow"  # price too high
    BGM_PRESENT = "bgm_present"                 # stone has BGM not captured
    MAKE_QUALITY = "make_quality"               # superior/inferior cut not reflected
    MARKET_MOVED = "market_moved"               # market shifted since the data
    SPECIAL_SITUATION = "special_situation"     # urgent/memo/special buyer
    DATA_ERROR = "data_error"                   # wrong input attributes
    RARE_ITEM = "rare_item"                     # rare shape/size needs manual call
    OTHER = "other"


# CLIENT RULE (2026-07): a reason is only demanded when the desk's price differs
# MATERIALLY from ours. A small gap is ordinary trading judgement — negotiation, a
# rush, a favoured buyer — and forcing a reason code on it just trains the desk to
# pick a junk value to clear the form, which poisons the reason analytics.
#
# 2.0 points — set by the client (2026-07). Their GS DIFF triage tabs already split
# at "less than 2", so a gap under 2 pts is the band they themselves treat as
# ordinary trading judgement. Above it, the suggestion is flagged for attention and
# the reason is required so the model learns WHY, not just that it was wrong.
VARIANCE_REASON_THRESHOLD_PTS: float = 2.0


def variance_pts(suggested_discount: float, human_discount: float | None) -> float | None:
    """|our discount - the desk's|, in points. None when there is no human price."""
    if human_discount is None:
        return None
    return abs(float(human_discount) - float(suggested_discount))


def needs_attention(suggested_discount: float, human_discount: float | None,
                    threshold: float = VARIANCE_REASON_THRESHOLD_PTS) -> bool:
    """True when the gap is big enough that the desk should be asked to look."""
    v = variance_pts(suggested_discount, human_discount)
    return v is not None and v > threshold


@dataclass
class FeedbackRecord:
    stone_id: str
    decision: str
    suggested_discount: float
    suggested_net: float
    # Stone attributes needed to turn the decision into a training example.
    shape_full: str
    weight: float
    color: str
    clarity: str
    cps: str = "NA"
    fluorescence: str = "Non"
    lab: str = "GIA"
    location: str = "NA"
    rap: float = 0.0
    # Human input.
    reason_code: str | None = None
    note: str = ""
    human_discount: float | None = None   # required for OVERRIDE (gold label)
    user: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def variance(self) -> float | None:
        """How far the desk moved our price, in discount points."""
        return variance_pts(self.suggested_discount, self.human_discount)

    @property
    def needs_attention(self) -> bool:
        """Large variance — the desk should be shown WHY we priced it this way."""
        return needs_attention(self.suggested_discount, self.human_discount)

    def validate(self) -> None:
        """Reject only what makes a record USELESS, never what makes it awkward.

        An OVERRIDE without a price carries no label, so it is still refused. A
        reason code, though, is only demanded on a materially different price (see
        VARIANCE_REASON_THRESHOLD_PTS): a small gap is normal trading judgement, and
        demanding a code for it produces junk codes, not insight.
        """
        if self.decision == Decision.OVERRIDE.value and self.human_discount is None:
            raise ValueError("An OVERRIDE requires the human's corrected discount.")
        # A REJECT carries no replacement price, so its reason IS the whole signal.
        if self.decision == Decision.REJECT.value and not self.reason_code:
            raise ValueError("A rejected suggestion requires a reason_code "
                             "(there is no price to learn from otherwise).")
        if (self.decision == Decision.OVERRIDE.value and self.needs_attention
                and not self.reason_code):
            raise ValueError(
                f"This override moves the price by {self.variance:.1f} pts "
                f"(> {VARIANCE_REASON_THRESHOLD_PTS} pt threshold) — a reason_code is "
                "required so the model learns WHY, not just that it was wrong.")


def record(fb: FeedbackRecord, path: Path | None = None) -> Path:
    """Append one decision to the immutable JSONL log."""
    fb.validate()
    p = path or FEEDBACK_LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(fb)) + "\n")
    # Mirror into the durable store. The JSONL stays the training pipeline's
    # source (unchanged, works offline); the database is the system of record for
    # concurrent live writes and for querying history — an append-only text file
    # has no transaction, and the CRM and the nightly job will write at once.
    try:
        from ..store import db
        db.record_decision(rec=asdict(fb), variance=fb.variance,
                           needs_attention=fb.needs_attention,
                           trainable=fb.human_discount is not None)
    except Exception:
        log.exception("decision written to JSONL but NOT to the store")
    log.info("Recorded %s for stone %s (reason=%s)", fb.decision, fb.stone_id, fb.reason_code)
    return p


def load_all(path: Path | None = None) -> list[dict]:
    p = path or FEEDBACK_LOG
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
