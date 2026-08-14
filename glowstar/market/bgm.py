"""BGM-as-base pricing (client request).

The client asked to "take BGM as base." We implement the rigorous reading the
trade uses: the BASE / reference price is a **No-BGM (clean) stone** — no brown/
green tinge, not milky. Any BGM present is an explicit DEDUCTION from that clean
base, learned from market data (market/aggregate_bulk -> bgm_discounts.json).

Three states, each handled explicitly (never silently):
  * clean       — milky/shade assessed and absent -> price at clean base, no cut.
  * bgm         — milky/shade present -> deduct the market-learned discount.
  * unassessed  — no BGM data on the stone (the client's own records have none
                  yet) -> price at the clean base but FLAG `bgm_unassessed` and
                  state plainly that the price ASSUMES no BGM. If the stone is in
                  fact BGM, it is over-priced — surfaced for human/CRM capture.

This is also why the client should start capturing milky/shade/eye-clean in the
CRM: it turns `unassessed` into `clean`/`bgm` and removes the assumption risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .aggregate_bulk import _milky_severity, _shade_class

_MILKY_NAME = {0: "none", 1: "light", 2: "medium", 3: "heavy"}
_BROWN_NAME = {0: "none", 1: "light", 2: "medium", 3: "heavy"}


def _ord(v) -> int | None:
    """A BGM ordinal as int, or None if missing/NaN."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else int(round(f))


@dataclass
class BgmAssessment:
    state: str                 # "clean" | "bgm" | "unassessed"
    milky_severity: str        # none/slight/medium/heavy
    shade_class: str           # none/neutral/negative/positive
    deduction_pts: float       # <= 0, applied to the clean-base discount
    assumes_no_bgm: bool       # True when unassessed (base assumed clean)
    note: str

    def as_dict(self) -> dict:
        return self.__dict__


def _present(v) -> bool:
    """True if `v` carries an actual assessment. NaN counts as ABSENT — it is what a
    missing cell reads as once the stone comes from a DataFrame, and `not in (None,
    "")` lets it through (NaN is truthy), which then misreports an unassessed stone
    as assessed and, downstream, crashed on .strip()."""
    if v is None:
        return False
    if isinstance(v, float) and v != v:
        return False
    return str(v).strip().lower() not in ("", "nan", "none")


def _has_bgm_fields(stone: dict) -> bool:
    return any(_present(stone.get(k)) for k in ("milky", "Milky", "shade",
                                                "Shade", "shade_name", "is_bgm"))


def assess(stone: dict, tables) -> BgmAssessment:
    """Classify a stone's BGM state and compute the deduction from the clean base.

    `tables` is a MarketTables (exposes soft_delta learned from market data).
    """
    # Preferred path: the client's live BgmComments, parsed to ordinals by the
    # loader. BGM is now a MODEL feature (learned from their own realized sales),
    # so we report the assessed STATE but apply NO post-model deduction here
    # (that would double-count what the model already prices).
    milky_o, brown_o = _ord(stone.get("milky_ord")), _ord(stone.get("brown_ord"))
    if milky_o is not None or brown_o is not None:
        # `or 0` turned an UNASSESSED None into 0.0 — the value that means
        # "assessed and clean". A stone with brown graded and milky missing came
        # back state="clean", assumes_no_bgm=False, and told the desk
        # "Assessed No-BGM from your inventory" about a field nobody had looked
        # at. No price moved (the deduction is 0.0 either way), but it is the
        # exact 0.0-vs-unassessed conflation Trap 4 exists to prevent, and it
        # states something untrue to the person deciding.
        partial = milky_o is None or brown_o is None
        m, b = milky_o or 0, brown_o or 0
        if m == 0 and b == 0 and partial:
            return BgmAssessment(
                state="partial", milky_severity="none", shade_class="none",
                deduction_pts=0.0, assumes_no_bgm=True,
                note="Partially assessed: "
                     + ("milky" if milky_o is None else "brown")
                     + " was not graded in your inventory. Priced at the clean "
                       "base, but this is NOT a confirmed No-BGM stone.")
        if m == 0 and b == 0:
            return BgmAssessment(
                state="clean", milky_severity="none", shade_class="none",
                deduction_pts=0.0, assumes_no_bgm=False,
                note="Assessed No-BGM (No Brown, No Milky) from your inventory — priced at the clean base.")
        return BgmAssessment(
            state="bgm", milky_severity=_MILKY_NAME.get(m, "?"),
            shade_class=("brown" if b > 0 else "none"), deduction_pts=0.0,
            assumes_no_bgm=False,
            note=f"BGM assessed from your inventory (milky={_MILKY_NAME.get(m,'?')}, "
                 f"brown={_BROWN_NAME.get(b,'?')}); priced by the model, which learned the "
                 "milky/brown discount from your own realized sales.")

    if not _has_bgm_fields(stone):
        return BgmAssessment(
            state="unassessed", milky_severity="unknown", shade_class="unknown",
            deduction_pts=0.0, assumes_no_bgm=True,
            note="No BGM data on this stone — price ASSUMES No-BGM (clean) base. "
                 "If the stone has brown/green tinge or milkiness it is over-priced; "
                 "capture milky/shade in the CRM to remove this assumption.",
        )

    milky_raw = stone.get("milky") or stone.get("Milky")
    shade_raw = stone.get("shade") or stone.get("Shade") or stone.get("shade_name")
    m = _milky_severity(milky_raw)
    s = _shade_class(shade_raw)
    deduction = tables.soft_delta(milky_raw, shade_raw) if tables is not None else 0.0

    if m == "none" and s in ("none", "neutral", "positive") and deduction == 0.0:
        return BgmAssessment(
            state="clean", milky_severity=m, shade_class=s, deduction_pts=0.0,
            assumes_no_bgm=False, note="Assessed No-BGM (clean) — priced at the clean base.",
        )
    return BgmAssessment(
        state="bgm", milky_severity=m, shade_class=s, deduction_pts=round(deduction, 2),
        assumes_no_bgm=False,
        note=f"BGM present (milky={m}, shade={s}); {deduction:.1f} pts deducted "
             "from the clean base, from market-learned BGM discounts.",
    )
