"""REST API — the two doors the client's CRM talks to.

    POST /price          price one stone
    POST /price/batch    price many in one call
    POST /decision       record what the desk did (accept / reject / override)
    GET  /health         is the service up, which model, how fresh is the data
    GET  /feedback/summary   what the desk has been telling us

WHY TWO DOORS
-------------
`/price` replaces the Excel round-trip: the CRM asks, the desk sees a price in
their own screen. `/decision` is the half that makes it a system rather than a
calculator — every accept, reject and override is stored immutably, so the engine
is measured against reality continuously instead of once per emailed workbook.

THE VARIANCE THRESHOLD (client rule, 2026-07)
---------------------------------------------
If the desk's price is within `VARIANCE_REASON_THRESHOLD_PTS` of ours, no reason
is required — that gap is ordinary trading judgement, and demanding a code for it
just trains the desk to click a junk value, which poisons the reason analytics.
Past the threshold the decision is flagged `needs_attention` and the reason is
required, so the model learns WHY it was wrong and not merely that it was.

Run:  uvicorn glowstar.service.app:app --host 0.0.0.0 --port 8000
FastAPI is optional; the pricing logic is fully usable without a server.
"""

from __future__ import annotations

import logging
import os

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - server is optional
    FastAPI = None

from .pricing_service import PricingService, StoneIn

log = logging.getLogger(__name__)

if FastAPI is not None:

    class DecisionIn(BaseModel):
        """What the desk did with a suggestion.

        `human_discount` is the desk's own discount off Rap (negative, e.g. -52.5).
        Send `human_ppc` instead if the CRM works in dollars per carat — the
        service converts it, so the CRM never has to do pricing arithmetic.
        """

        stone_id: str
        decision: str = Field(description="accept | reject | override")
        suggested_discount: float
        suggested_net: float = 0.0
        # the stone, so the decision can become a training example later
        shape_full: str = "NA"
        weight: float = 0.0
        color: str = "NA"
        clarity: str = "NA"
        cps: str = "NA"
        fluorescence: str = "Non"
        lab: str = "GIA"
        location: str = "NA"
        rap: float = 0.0
        # the human's input
        human_discount: float | None = None
        human_ppc: float | None = None
        reason_code: str | None = None
        note: str = ""
        user: str = ""

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(_app):
        """Fail fast at BOOT, not on the desk's first call."""
        if os.environ.get("GS_ENV", "").lower() in ("prod", "production") \
                and not os.environ.get("GS_API_KEY"):
            raise RuntimeError(
                "GS_ENV=production but GS_API_KEY is not set — refusing to serve "
                "prices on an unauthenticated endpoint.")
        _get_service()          # load the model now; a cold first request is a bug
        yield

    app = FastAPI(
        title="Glow Star Pricing Engine",
        version="1.0.0",
        description="Prices a certified natural polished stone as a discount off "
                    "the Rapaport list, and records the desk's decision so the "
                    "engine is measured against reality continuously.",
        lifespan=_lifespan,
    )
    _service: PricingService | None = None

    def _get_service() -> PricingService:
        global _service
        if _service is None:
            # use_feedback stays OFF: never replay the desk's own corrections back
            # at them (CLAUDE.md Trap 2).
            _service = PricingService()
        return _service

    def require_key(x_api_key: str = Header(default="")) -> None:
        """Shared-secret check. Set GS_API_KEY in the environment to enable it.

        Unset = open, which is fine on a private network but must NOT be how this
        is exposed to anything routable. The server refuses to start unprotected
        in production (see `_startup`).
        """
        expected = os.environ.get("GS_API_KEY", "")
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> dict:
        """Liveness + WHAT is actually serving. A 200 that hides a stale model is
        not health, so the model version and data age are part of the answer."""
        from ..models import registry
        from ..config import PATHS
        import datetime as _dt

        out: dict = {"status": "ok"}
        try:
            _, card = registry.load_current()
            out["model"] = {"version": (card or {}).get("version"),
                            "trained_at": (card or {}).get("trained_at"),
                            "test_mae": (card or {}).get("test_mae")}
        except Exception as e:
            out["status"] = "degraded"
            out["model"] = {"error": type(e).__name__}
        try:
            age_h = (_dt.datetime.now().timestamp() - PATHS.records.stat().st_mtime) / 3600
            out["records_age_hours"] = round(age_h, 1)
            if age_h > 48:
                out["status"] = "degraded"
                out["warning"] = "inventory data is more than 48h old"
        except OSError:
            out["status"] = "degraded"
            out["records_age_hours"] = None
        return out

    @app.post("/price", dependencies=[Depends(require_key)])
    def price(stone: StoneIn) -> dict:
        """Price one stone: suggestion, interval, comparables, flags, explanation."""
        return _get_service().price(stone)

    @app.post("/price/batch", dependencies=[Depends(require_key)])
    def price_batch(stones: list[StoneIn]) -> dict:
        """Price many stones in one call.

        One bad stone must never fail the whole book, so each is priced
        independently and failures come back per stone with their reason.
        """
        if len(stones) > 5000:
            raise HTTPException(status_code=413, detail="max 5000 stones per call")
        svc = _get_service()
        priced, failed = [], []
        for s in stones:
            try:
                priced.append(svc.price(s))
            except Exception as e:
                log.exception("pricing failed for %s", s.StoneId)
                failed.append({"stone_id": s.StoneId, "error": f"{type(e).__name__}: {e}"})
        return {"priced": priced, "failed": failed,
                "n_priced": len(priced), "n_failed": len(failed)}

    @app.post("/decision", dependencies=[Depends(require_key)])
    def decision(d: DecisionIn) -> dict:
        """Record what the desk did. This is what makes the loop a loop.

        Returns the measured variance and whether it needs attention, so the CRM
        can show the desk the same threshold the engine uses — one rule, one place.
        """
        from ..feedback.store import (Decision, FeedbackRecord, record,
                                      VARIANCE_REASON_THRESHOLD_PTS,
                                      needs_attention, variance_pts)

        if d.decision not in {x.value for x in Decision}:
            raise HTTPException(status_code=422,
                                detail=f"decision must be one of {[x.value for x in Decision]}")

        human = d.human_discount
        if human is None and d.human_ppc is not None:
            if d.rap <= 0:
                raise HTTPException(status_code=422,
                                    detail="human_ppc needs rap > 0 to convert to a discount")
            human = round(d.human_ppc / d.rap * 100.0 - 100.0, 2)

        rec = FeedbackRecord(
            stone_id=d.stone_id, decision=d.decision,
            suggested_discount=d.suggested_discount, suggested_net=d.suggested_net,
            shape_full=d.shape_full, weight=d.weight, color=d.color, clarity=d.clarity,
            cps=d.cps, fluorescence=d.fluorescence, lab=d.lab, location=d.location,
            rap=d.rap, reason_code=d.reason_code, note=d.note,
            human_discount=human, user=d.user,
        )
        try:
            record(rec)                       # validate() runs inside; 422 on a bad record
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None

        # Online corrections refresh immediately so a repeated mistake shifts the
        # next quote without waiting for the nightly retrain.
        try:
            from ..feedback.learning import build_corrections
            from ..feedback import store as fbstore
            _get_service().engine.set_corrections(build_corrections(fbstore.load_all()))
        except Exception:
            log.exception("could not refresh online corrections (decision still stored)")

        v = variance_pts(d.suggested_discount, human)
        return {
            "recorded": True,
            "stone_id": d.stone_id,
            "variance_pts": None if v is None else round(v, 2),
            "threshold_pts": VARIANCE_REASON_THRESHOLD_PTS,
            "needs_attention": needs_attention(d.suggested_discount, human),
            "reason_required": needs_attention(d.suggested_discount, human),
        }

    @app.get("/feedback/summary", dependencies=[Depends(require_key)])
    def feedback_summary() -> dict:
        """What the desk has been telling us — acceptance rate and reason mix."""
        from ..feedback import store as fbstore
        from ..feedback.learning import reason_summary
        return reason_summary(fbstore.load_all())

else:  # pragma: no cover
    app = None
