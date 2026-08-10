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
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - server is optional
    FastAPI = None

from . import ratelimit
from .pricing_service import PricingService, StoneIn
from .frontoffice import (FrontOfficeStone, FrontOfficeReason,
                          MasterDiscountRequest)

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

    def require_key(request: Request, x_api_key: str = Header(default="")) -> None:
        """Shared-secret check, then a rate limit. Attached to every priced route.

        Set GS_API_KEY in the environment to enable the key check. Unset = open,
        which is fine on a private network but must NOT be how this is exposed to
        anything routable. The server refuses to start unprotected in production
        (see `_startup`).

        The rate limit runs AFTER the key check on purpose: a caller with a bad
        key should not be able to consume a legitimate caller's allowance, and
        rejecting on the key first is the cheaper of the two.

        `/health` does not depend on this — a monitor must be able to probe
        liveness without spending quota, and it exposes nothing.
        """
        expected = os.environ.get("GS_API_KEY", "")
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

        # Per key AND per source address: one runaway CRM instance must not
        # starve the desk's other machines sharing the same key.
        client = request.client.host if request.client else "-"
        caller = f"{x_api_key[:8]}|{request.headers.get('x-real-ip') or client}"
        ok, remaining, retry_after = ratelimit.check(caller)
        if not ok:
            log.warning("rate limit hit by %s — retry in %.0fs", caller, retry_after)
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded; retry in {retry_after:.0f}s",
                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
            )
        request.state.rate_remaining = remaining

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
        # Surface the store: an API that is "ok" while silently failing to record
        # anything is the failure mode this whole layer exists to prevent.
        try:
            from ..store import db
            out["store"] = db.counts()
            if "error" in out["store"]:
                out["status"] = "degraded"
        except Exception as e:
            out["status"] = "degraded"
            out["store"] = {"error": type(e).__name__}
        return out

    def _audit(res: dict, stone: StoneIn, source: str) -> dict:
        """Write the published price to the audit trail.

        Every price we publish must be answerable later — "what did you quote in
        March, and which model said it?" is the first question asked when a price
        is disputed, and the MOU requires the trail. Only `/frontoffice/price`
        recorded quotes; `/price` and `/price/batch` published prices that left
        no record at all, so anything the desk priced through those endpoints was
        invisible afterwards.

        `record_quote` is best-effort and never raises (glowstar/store/db.py), so
        a dead database costs an audit row, never the price itself.
        """
        try:
            from ..store import db
            from ..models import registry
            db.record_quote(facts=res.get("suggestion", {}),
                            stone=stone.model_dump(),
                            model_version=registry.current_version(),
                            source=source)
        except Exception:
            log.exception("audit write failed — the price was still served")
        return res

    @app.post("/price", dependencies=[Depends(require_key)])
    def price(stone: StoneIn) -> dict:
        """Price one stone: suggestion, interval, comparables, flags, explanation."""
        return _audit(_get_service().price(stone), stone, "api")

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
                priced.append(_audit(svc.price(s), s, "api-batch"))
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

    # ----------------------------------------------------------------------
    # FrontOffice contract (their spec, 29-07-2026) — their field names, their
    # response shape, so their CRM binds to it without a translation layer.
    # ----------------------------------------------------------------------
    @app.post("/frontoffice/price", dependencies=[Depends(require_key)])
    def fo_price(stones: list[dict]) -> list:
        """Spec #1 — bulk stone pricing. One response row per stone.

        Takes raw dicts rather than a typed list on purpose: FastAPI validates a
        typed list all-or-nothing, and one malformed row must never cost the desk
        the other 4,999 prices. Each row is validated individually below.
        """
        from .frontoffice import price_stones
        if len(stones) > 5000:
            raise HTTPException(status_code=413, detail="max 5000 stones per call")
        parsed, bad = [], []
        for i, raw in enumerate(stones):
            try:
                parsed.append(FrontOfficeStone(**raw))
            except Exception as e:
                # A malformed row is reported in place; it never fails the book.
                bad.append({"StoneId": (raw or {}).get("stoneId"), "index": i,
                            "AIDiscount": None, "Error": f"invalid stone: {e}"})
        return price_stones(parsed, _get_service()) + bad

    @app.post("/frontoffice/reason", dependencies=[Depends(require_key)])
    def fo_reason(fr: FrontOfficeReason) -> dict:
        """Spec #2 — the desk's reason for a stone, by certificate number.

        Their document sends certificateNo + reason + aiDiscount but NOT the
        desk's own price. A reason with no corrected price records THAT we were
        wrong, never WHAT right looks like — there is no label to train on. So
        `deskDiscount`/`deskPpc` is accepted and, when present, is what makes the
        record trainable. Without it the reason is stored as analytics only, and
        the response says so rather than implying the model learned something.
        """
        from ..feedback.store import (Decision, FeedbackRecord, record,
                                      VARIANCE_REASON_THRESHOLD_PTS,
                                      needs_attention, variance_pts)
        desk = fr.deskDiscount
        if desk is None and fr.deskPpc is not None:
            raise HTTPException(
                status_code=422,
                detail="deskPpc needs the stone's Rap to convert; send deskDiscount "
                       "or include rap")
        rec = FeedbackRecord(
            stone_id=fr.stoneId or fr.certificateNo,
            decision=(Decision.OVERRIDE.value if desk is not None else Decision.REJECT.value),
            suggested_discount=fr.aiDiscount, suggested_net=0.0,
            shape_full="NA", weight=0.0, color="NA", clarity="NA",
            reason_code=fr.reason, note=f"certificateNo={fr.certificateNo}",
            human_discount=desk, user=fr.user or "",
        )
        try:
            record(rec)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None

        v = variance_pts(fr.aiDiscount, desk)
        return {
            "recorded": True,
            "certificateNo": fr.certificateNo,
            "trainable": desk is not None,
            "note": ("stored as a training label" if desk is not None else
                     "stored for ANALYTICS ONLY — send deskDiscount (the desk's own "
                     "price) to make this a training label"),
            "variance_pts": None if v is None else round(v, 2),
            "threshold_pts": VARIANCE_REASON_THRESHOLD_PTS,
            "needs_attention": needs_attention(fr.aiDiscount, desk),
        }

    @app.post("/frontoffice/master-discount", dependencies=[Depends(require_key)])
    def fo_master_discount(m: MasterDiscountRequest) -> dict:
        """Spec #3 — price a GRID CELL (a weight range), not one stone.

        Answered at the MIDPOINT of the requested weight range, which is what a
        cell-level discount means. The support behind it is returned too: a cell
        the engine cannot back with data must not read like one it can.
        """
        from .frontoffice import to_stone_in
        from .tradeability import tradeability_for
        if m.toWeight < m.fromWeight:
            raise HTTPException(status_code=422, detail="toWeight must be >= fromWeight")

        mid = (m.fromWeight + m.toWeight) / 2.0
        stone = FrontOfficeStone(
            stoneId=f"CELL-{m.fromWeight}-{m.toWeight}", shape=m.shape, weight=mid,
            color=m.color, clarity=m.clarity, cps=m.cps or "3EX",
            fluorescence=m.floro or "Non", lab="GIA")
        try:
            res = _get_service().price(to_stone_in(stone))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"cannot price this cell: {e}") from None
        f = res["suggestion"]
        tr = tradeability_for(m.shape, mid, m.color, m.clarity)
        return {
            "fromWeight": m.fromWeight, "toWeight": m.toWeight,
            "pricedAtWeight": round(mid, 2),
            "color": m.color, "clarity": m.clarity,
            "cps": m.cps, "floro": m.floro, "shape": m.shape,
            "AIDiscount": f["suggested_discount"],
            "AIPricePerCarat": f["suggested_ppc"],
            "FairRangeLow": f["ci_discount_low"],
            "FairRangeHigh": f["ci_discount_high"],
            "MarketComparables": f.get("comparable_count"),
            "Tradeability": tr["label"], "TradeabilityDays": tr["median_days"],
            "Method": f.get("method"),
            "Note": "cell-level discount, priced at the midpoint of the weight range",
        }

    @app.get("/feedback/summary", dependencies=[Depends(require_key)])
    def feedback_summary() -> dict:
        """What the desk has been telling us — acceptance rate and reason mix."""
        from ..feedback import store as fbstore
        from ..feedback.learning import reason_summary
        return reason_summary(fbstore.load_all())

else:  # pragma: no cover
    app = None
