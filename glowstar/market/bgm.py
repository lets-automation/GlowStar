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

from dataclasses import dataclass

from .aggregate_bulk import _milky_severity, _shade_class


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


def _has_bgm_fields(stone: dict) -> bool:
    return any(stone.get(k) not in (None, "") for k in ("milky", "Milky", "shade",
                                                        "Shade", "shade_name", "is_bgm"))


def assess(stone: dict, tables) -> BgmAssessment:
    """Classify a stone's BGM state and compute the deduction from the clean base.

    `tables` is a MarketTables (exposes soft_delta learned from market data).
    """
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
