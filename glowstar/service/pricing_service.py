"""PricingService — fit once, price many (brief Section 6, service API).

Validates an incoming stone with pydantic, runs it through the trained
PricingEngine, attaches the guarded narration, and returns a clean JSON-able
dict. This is the callable the REST layer (service/app.py) wraps.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
from pydantic import BaseModel, Field

from ..data.loaders import load_records, sold_stones
from ..models.engine import PricingEngine, EngineConfig
from ..narration.narrate import narrate
from ..feedback import store as fbstore
from ..feedback.learning import build_corrections, reason_summary


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
    def __init__(self, engine: PricingEngine | None = None, use_feedback: bool = True):
        self._feedback = fbstore.load_all() if use_feedback else []
        if engine is None:
            df, _ = load_records()
            sold = sold_stones(df, drop_outliers=True)
            # Production model: train on ALL sold history + human feedback labels.
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
        out = {"suggestion": facts, "market": self._market_context()}
        if explain:
            out["explanation"] = narrate(facts)
        return out

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
