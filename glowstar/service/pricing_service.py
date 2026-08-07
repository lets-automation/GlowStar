"""PricingService — fit once, price many (brief Section 6, service API).

Validates an incoming stone with pydantic, runs it through the trained
PricingEngine, attaches the guarded narration, and returns a clean JSON-able
dict. This is the callable the REST layer (service/app.py) wraps.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from ..data.loaders import load_records, sold_stones
from ..models.engine import PricingEngine, EngineConfig
from ..models import registry
from ..narration.narrate import narrate
from ..feedback import store as fbstore
from ..feedback.learning import build_corrections, reason_summary
from ..reference.rap_versioning import RapChangeMonitor

log = logging.getLogger(__name__)

# How much to widen the confidence band (each side, discount points) while a
# stone's Rap cell is inside the post-change adjustment window — the market level
# is genuinely less certain there, so the band must say so.
_RAP_CHANGE_CI_WIDEN_PTS = 4.0


class StoneIn(BaseModel):
    """A stone to price. Only pricing-time attributes; no transaction fields.

    Deliberately small: the CALLER should have to supply the five things that
    identify a stone, and nothing that we can determine ourselves or that they
    could get wrong.
    """

    StoneId: str = ""
    # Canonicalised on the way IN — see the validator below. This is the single
    # boundary every API caller crosses (/price, /price/batch and the FrontOffice
    # endpoints all build a StoneIn), so fixing it here fixes all of them at once.
    Shape_full: str
    Weight: float = Field(gt=0)
    Color: str
    Clarity: str
    CPS: str = "NA"
    Fluorescence: str = "Non"
    Lab: str = "GIA"
    Location: str = "NA"
    # Rap is OPTIONAL — we look it up from the licensed sheet when it is absent.
    #
    # It used to be required, which quietly made the caller responsible for the
    # yardstick every price is measured against. That is the single worst field to
    # delegate: a CRM holding a stale Rap would shift EVERY discount we return, and
    # nothing in the response would look wrong. It is also exactly the failure that
    # already bit this project once (their sheet re-based 0.30-0.39 rounds ~+7% and
    # our $/ct went stale for that band alone).
    #
    # Our sheet is verified against the client's own book at 100% exact on recent
    # sales, so looking it up is both safer and less work for them. A caller may
    # still pass Rap to override — e.g. to reproduce a historical quote.
    Rap: float | None = Field(default=None, gt=0)
    # Measurements — enable the spread/face-up premium on rounds (market/spread.py).
    Length: float | None = None
    Width: float | None = None
    Depth: float | None = None
    # Tinge, as the client's inventory API now supplies it (Brown/Milky/Shade/Green).
    Brown: str | None = None
    Milky: str | None = None
    Shade: str | None = None
    Green: str | None = None
    # Legacy aliases kept so an older caller does not break.
    milky: str | None = None

    @field_validator("Shape_full")
    @classmethod
    def _canon_shape(cls, v: str) -> str:
        """Accept the trade code the client's systems actually send.

        Their inventory API sends `RBC` / `OB` / `PB` / `MB`, and a CRM field may
        arrive as `ROUND` or `round`. The engine routes on an exact-string lookup
        against `Shape_full` as TRAINED ("Round"), so anything else scored zero
        training rows, was flagged `rare_shape` and fell to the sparse fallback —
        6.1 points deeper on a 1.01 G VS1. Verified end-to-end over HTTP, not
        reasoned about.
        """
        from ..reference.normalize import normalize_shape
        return normalize_shape(v) or v

    @field_validator("CPS")
    @classmethod
    def _canon_cps(cls, v: str) -> str:
        """Same defect as the shape field, on cut/polish/symmetry.

        The model's cut vocabulary is closed (3EX/EX/VG/GD/FR/PR). This endpoint
        took CPS as a free string, so "3EX", "EX-EX-EX" and "EX EX EX" — one
        stone, three spellings — returned -45.85, -51.44 and -59.54. The last is
        what you get for sending no cut information at all.
        """
        from ..reference.normalize import normalize_cps
        return normalize_cps(v)


class PricingService:
    # DEFAULT use_feedback=False. It used to default to True, which is the
    # override-echo trap (CLAUDE.md Trap 2): with feedback on, re-pricing a stone
    # the desk has already corrected replays THEIR OWN number back at them. Every
    # reviewed stone then lands on target to the decimal — it looks like a triumph,
    # measures nothing, and collapses on the next unseen file.
    #
    # It matters far more now than it did as a batch script: once the CRM calls
    # this service live, this is THE pricing path, and the desk re-prices stones
    # they have already touched all day long.
    #
    # Feedback is still LOADED (for segment analytics and the reason summary); it
    # simply never rewrites a price unless a caller explicitly opts in.
    def __init__(self, engine: PricingEngine | None = None, use_feedback: bool = False,
                 prefer_registry: bool = True, rap_monitor: RapChangeMonitor | None = None):
        self._feedback = fbstore.load_all() if use_feedback else []
        # Rap-change "red line": flags a stone whose Rap cell just moved and widens
        # its band during the adjustment window. Inert until >=2 list versions are
        # ingested (see reference.rap_versioning) — never guesses a change.
        self._rap_monitor = rap_monitor if rap_monitor is not None else RapChangeMonitor()
        # Prefer a gated, versioned model from the registry (fast start, audited)
        # over retraining in memory. The nightly retrain job promotes it.
        if engine is None and prefer_registry:
            engine, card = registry.load_current()
            if engine is not None:
                log.info("Loaded live model %s from registry (test MAE=%s).",
                         (card or {}).get("version"), (card or {}).get("test_mae"))
        if engine is None:
            df, _ = load_records()
            sold = sold_stones(df, drop_outliers=True)
            # Cold start: train on ALL sold history + human feedback labels.
            engine = PricingEngine(EngineConfig()).fit(sold, feedback_records=self._feedback)
        self.engine = engine
        self._asof = engine._train_max_date

    def _to_frame(self, stone: StoneIn) -> pd.DataFrame:
        row = stone.model_dump()
        row["Shape"] = None
        # Price "as of now": use the most recent known date as the market clock.
        row["MarketSheetDate_dt"] = self._asof
        row["OrderDate_dt"] = self._asof

        # Rap: look it up unless the caller deliberately supplied one. Doing this
        # here (not in the caller's CRM) keeps one licensed sheet as the single
        # yardstick — see StoneIn.Rap.
        if not row.get("Rap"):
            from ..reference import rap_lookup as RL
            res = RL.lookup(shape_code=None, shape_full=stone.Shape_full,
                            weight=stone.Weight, color=stone.Color, clarity=stone.Clarity)
            if res.ok:
                row["Rap"] = res.price_per_ct
                row["Rap_status"] = "ok"
            elif res.floor_estimate:
                # A labelled estimate, never silently passed off as published.
                row["Rap"] = res.floor_estimate
                row["Rap_status"] = res.status.value
            else:
                raise ValueError(
                    f"No Rapaport price for {stone.Weight}ct {stone.Color}/{stone.Clarity} "
                    f"{stone.Shape_full} ({res.status.value}): {res.note}")

        # Tinge -> the ordinals the model was trained on. Accept the client's
        # structured codes (LBR/MML/HMT/...) exactly as their inventory API emits
        # them, so the CRM forwards its own field values with no translation.
        from ..data.loaders import parse_tinge
        for src, suffix, col in (("Brown", "BR", "brown_ord"), ("Milky", "ML", "milky_ord"),
                                 ("Shade", "MT", "shade_ord"), ("Green", "GR", "green_ord")):
            raw = row.get(src)
            row[col] = parse_tinge(raw, suffix) if raw is not None else float("nan")
        return pd.DataFrame([row])

    def price(self, stone: StoneIn, *, explain: bool = True) -> dict:
        df = self._to_frame(stone)
        suggestion = self.engine.predict(df)[0]
        facts = asdict(suggestion)
        facts["coverage_pct"] = int(self.engine.cfg.coverage * 100)
        rap_change = self._apply_rap_change(facts, stone)
        out = {"suggestion": facts, "market": self._market_context()}
        if rap_change is not None:
            out["rap_change"] = rap_change
        if explain:
            out["explanation"] = narrate(facts)
        return out

    def _apply_rap_change(self, facts: dict, stone: StoneIn) -> dict | None:
        """If the stone's Rap cell recently moved, attach the red-line info and,
        while inside the adjustment window, widen the band and flag it. The
        suggested DISCOUNT is unchanged (it already applies to the current Rap);
        only the stated uncertainty grows, honestly, until the level re-settles."""
        info = self._rap_monitor.check(stone.Shape_full, stone.Weight, stone.Color, stone.Clarity)
        if not info.changed:
            return None
        if info.in_window:
            w = _RAP_CHANGE_CI_WIDEN_PTS
            facts["ci_discount_low"] = round(facts["ci_discount_low"] - w, 2)
            facts["ci_discount_high"] = round(facts["ci_discount_high"] + w, 2)
            rap, wt = stone.Rap, stone.Weight
            facts["ci_net_low"] = round(rap * (1 + facts["ci_discount_low"] / 100.0) * wt, 2)
            facts["ci_net_high"] = round(rap * (1 + facts["ci_discount_high"] / 100.0) * wt, 2)
            facts["flags"] = sorted(set(facts.get("flags", []) + ["rap_recently_changed"]))
        return info.as_dict()

    def _market_context(self, include_macro: bool = False) -> dict:
        """Market context for a price. By default only REAL, data-derived signals
        are returned (the internal trend computed from the client's own sales).

        The external macro reference (RAPI/lab-grown/tariffs) is a SEEDED set of
        sourced facts, not a live feed, so it is OFF by default and only included
        when explicitly requested (include_macro=True) — clearly labelled.
        """
        internal = self.engine.index.as_dict() if self.engine.index is not None else {}
        ctx = {"internal_trend": internal}
        if include_macro:
            from ..market.context import current_context, cross_check
            ctx["macro_reference_seeded"] = current_context()
            ctx["cross_check"] = cross_check(internal.get("direction", "flat"))
        return ctx

    def price_payload(self, payload: dict, *, explain: bool = True) -> dict:
        return self.price(StoneIn(**payload), explain=explain)

    # --- human-in-the-loop feedback ---

    def record_decision(self, *, stone: StoneIn, decision: str,
                        suggested_discount: float, suggested_net: float,
                        reason_code: str | None = None, note: str = "",
                        human_discount: float | None = None, user: str = "") -> dict:
        """Record a pricer's accept/reject/override decision and learn from it.

        Rejections require a reason_code; overrides require the human's price.
        The decision is stored immutably and online corrections refresh at once
        (durable learning happens on the next retrain via fit(feedback_records=)).
        """
        rec = fbstore.FeedbackRecord(
            stone_id=stone.StoneId, decision=decision,
            suggested_discount=suggested_discount, suggested_net=suggested_net,
            shape_full=stone.Shape_full, weight=stone.Weight, color=stone.Color,
            clarity=stone.Clarity, cps=stone.CPS, fluorescence=stone.Fluorescence,
            lab=stone.Lab, location=stone.Location, rap=stone.Rap,
            reason_code=reason_code, note=note, human_discount=human_discount, user=user,
        )
        fbstore.record(rec)
        self._feedback = fbstore.load_all()
        self.engine.set_corrections(build_corrections(self._feedback))   # immediate effect
        return {"recorded": True, "feedback_summary": reason_summary(self._feedback)}
