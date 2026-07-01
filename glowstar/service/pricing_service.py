"""PricingService — fit once, price many (brief Section 6, service API).

Validates an incoming stone with pydantic, runs it through the trained
PricingEngine, attaches the guarded narration, and returns a clean JSON-able
dict. This is the callable the REST layer (service/app.py) wraps.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import pandas as pd
from pydantic import BaseModel, Field

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
    """A stone to price. Only pricing-time attributes; no transaction fields."""

    StoneId: str = ""
    Shape_full: str
    Weight: float = Field(gt=0)
    Color: str
    Clarity: str
    CPS: str = "NA"
    Fluorescence: str = "Non"
    Lab: str = "GIA"
    Location: str = "NA"
    Rap: float = Field(gt=0)
    # Optional soft attributes (market-learned adjustment applies when present).
    milky: str | None = None
    Shade: str | None = None


class PricingService:
    def __init__(self, engine: PricingEngine | None = None, use_feedback: bool = True,
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
