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

    def validate(self) -> None:
        if self.decision in (Decision.REJECT.value, Decision.OVERRIDE.value) and not self.reason_code:
            raise ValueError("A rejected/overridden suggestion requires a reason_code.")
        if self.decision == Decision.OVERRIDE.value and self.human_discount is None:
            raise ValueError("An OVERRIDE requires the human's corrected discount.")


def record(fb: FeedbackRecord, path: Path | None = None) -> Path:
    """Append one decision to the immutable JSONL log."""
    fb.validate()
    p = path or FEEDBACK_LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(fb)) + "\n")
    log.info("Recorded %s for stone %s (reason=%s)", fb.decision, fb.stone_id, fb.reason_code)
    return p


def load_all(path: Path | None = None) -> list[dict]:
    p = path or FEEDBACK_LOG
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
