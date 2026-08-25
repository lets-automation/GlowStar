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


def _grid_age_days() -> int | None:
    """Days since the newest edit in the point-in-time grid history.

    Reads the file's tail rather than parsing ~100 MB of JSON — health must stay
    fast enough to be polled. Returns None if there is no history at all.
    """
    import datetime as _dt
    import re as _re
    from ..config import PATHS

    p = getattr(PATHS, "grid_history", None)
    if p is None:
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "data" / "master_grid" / "history.json"
    if not p.exists():
        return None
    newest = ""
    pat = _re.compile(rb'"(20\d\d-\d\d-\d\d)T')
    with p.open("rb") as fh:
        size = p.stat().st_size
        # The store is keyed by cell, so dates are scattered — scan the last few
        # MB, which reliably contains recent edits without reading the whole file.
        fh.seek(max(0, size - (4 << 20)))
        for m in pat.finditer(fh.read()):
            d = m.group(1).decode()
            if d > newest:
                newest = d
    if not newest:
        return None
    return (_dt.date.today() - _dt.date.fromisoformat(newest)).days

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
        # EVERY problem, not just the last one found. Each check used to assign
        # out["warning"] directly, so a stale grid AND an old model reported only
        # whichever check ran last — an operator fixes the one they can see and
        # the other stays hidden. Accumulate, then publish all of them.
        warns: list[str] = []

        def _warn(msg: str) -> None:
            out["status"] = "degraded"
            warns.append(msg)

        try:
            _, card = registry.load_current()
            registry_version = (card or {}).get("version")
            out["model"] = {"version": registry_version,
                            "trained_at": (card or {}).get("trained_at"),
                            "test_mae": (card or {}).get("test_mae")}
            # WHAT THIS WORKER IS ACTUALLY SERVING.
            #
            # The model is loaded once per uvicorn worker and there is no reload
            # path, so a nightly promotion does NOT reach a running worker — only
            # a restart does. This endpoint read the REGISTRY and reported the
            # freshly promoted version as though it were live, which is the exact
            # failure CLAUDE.md Trap 5 describes: a health surface describing a
            # pipeline the client is not being served by. Every accuracy gain the
            # gate reported could sit on disk indefinitely, unseen, while /health
            # stayed green.
            #
            # `serving_version` is read off the loaded service, so the two can
            # disagree — and when they do, that IS the alarm.
            serving = getattr(_service, "model_version", None)
            out["model"]["serving_version"] = serving
            if _service is None:
                out["model"]["serving_version"] = "not-loaded"
            elif serving and registry_version and serving != registry_version:
                _warn(f"worker is serving model {serving} but {registry_version} "
                      "is promoted — restart glowstar-api to pick it up")
        except Exception as e:
            out["status"] = "degraded"
            out["model"] = {"error": type(e).__name__}
        try:
            age_h = (_dt.datetime.now().timestamp() - PATHS.records.stat().st_mtime) / 3600
            out["records_age_hours"] = round(age_h, 1)
            if age_h > 48:
                _warn("inventory data is more than 48h old")
        except OSError:
            out["status"] = "degraded"
            out["records_age_hours"] = None
        # GRID FRESHNESS — the thing CLAUDE.md calls "the whole ballgame", and
        # the single largest measured driver of pricing error (fresh cell ~2.0
        # MAE, 30d+ stale ~3.1).
        #
        # It was ABSENT from every health surface, and that is precisely how the
        # nightly jobs stayed dead for fifteen days without anyone noticing:
        # `records.json` kept refreshing, so this endpoint stayed green while the
        # grid rotted underneath it. Health that reports the healthy half of a
        # broken pipeline is worse than no health check at all.
        try:
            age_d = _grid_age_days()
            out["grid_age_days"] = age_d
            if age_d is None:
                _warn("no grid history on disk")
            elif age_d > 3:
                _warn(f"price grid is {age_d} days stale — the nightly job "
                      "may have stopped")
        except Exception:
            out["status"] = "degraded"
            out["grid_age_days"] = None

        # MODEL AGE. The promotion gate protects against a BAD model; nothing
        # protected against NO model. A retrain that dies before reaching the
        # gate leaves the incumbent serving forever and logs nothing loud.
        try:
            trained = (out.get("model") or {}).get("trained_at")
            if trained:
                days = (_dt.datetime.now() - _dt.datetime.fromisoformat(trained)).days
                out["model"]["age_days"] = days
                if days > 2:
                    _warn(f"model is {days} days old — the nightly retrain "
                          "has not promoted since then")
        except Exception:
            pass

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

        if warns:
            out["warnings"] = warns
            out["warning"] = "; ".join(warns)   # kept: existing consumers read this
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
        """Price one stone: suggestion, interval, comparables, flags, explanation.

        A stone we CANNOT price returns a structured 422, never a 500.

        Fancy-colour and cape stones ("Fancy Intense Yellow") have no cell on the
        white D-N Rapaport list, so the lookup raises. Unhandled, that surfaced in
        the client's CRM as a bare `[object Object]` dialog which aborted a whole
        595-stone run — 593 perfectly priceable stones lost to two yellows.
        A 422 with the stone named lets the CRM skip it and carry on.

        We deliberately do NOT invent a number for these. The model is trained on
        white goods priced off the white list; a fancy colour has neither, and a
        plausible-looking price for a stone we cannot price is worse than a clear
        refusal.
        """
        try:
            return _audit(_get_service().price(stone), stone, "api")
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail={"error": "not_priceable",
                        "stone_id": stone.StoneId or None,
                        "shape": stone.Shape_full, "weight": stone.Weight,
                        "color": stone.Color, "clarity": stone.Clarity,
                        "message": str(e),
                        "hint": "Fancy/cape colours have no white Rapaport cell. "
                                "Use /frontoffice/price for batches — it returns a "
                                "per-stone Error and never fails the whole book."},
            ) from None

    @app.post("/price/batch", dependencies=[Depends(require_key)])
    def price_batch(stones: list[StoneIn]) -> dict:
        """Price many stones in one call.

        One bad stone must never fail the whole book, so each is priced
        independently and failures come back per stone with their reason.
        """
        if len(stones) > 5000:
            raise HTTPException(status_code=413, detail="max 5000 stones per call")
        # ONE model call for the whole book — see PricingService.price_many.
        # The per-stone loop ran ~95 ms/stone, so the documented 5,000-stone limit
        # took ~8 minutes against nginx's 300 s timeout: the advertised maximum
        # request exceeded the advertised timeout. Batched it is ~2.3 ms/stone.
        svc = _get_service()
        priced, failed = [], []
        for s, res in zip(stones, svc.price_many(stones)):
            if isinstance(res, Exception):
                log.exception("pricing failed for %s: %s", s.StoneId, res)
                failed.append({"stone_id": s.StoneId,
                               "error": f"{type(res).__name__}: {res}"})
            else:
                priced.append(_audit(res, s, "api-batch"))
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
                                      is_trainable, needs_attention, variance_pts)

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
                                      is_trainable, needs_attention, variance_pts)
        desk = fr.deskDiscount
        if desk is None and fr.deskPpc is not None:
            raise HTTPException(
                status_code=422,
                detail="deskPpc needs the stone's Rap to convert; send deskDiscount "
                       "or include rap")
        # THE STONE MUST BE ATTACHED, or the record teaches nothing. Corrections
        # are learned per price cell, and a cell is shape/weight/colour/clarity.
        # This used to hardcode "NA"/0.0, which silently made every desk
        # correction untrainable — 14 of them, each carrying the desk's own
        # price, the hardest feedback in the world to obtain.
        # Order: what the caller sent, else look the stone up, else record it
        # honestly as unattributable.
        from .frontoffice import resolve_stone
        found = resolve_stone(fr.certificateNo, fr.stoneId) or {}
        shape = fr.shape or found.get("shape_full") or "NA"
        weight = fr.weight if fr.weight is not None else found.get("weight", 0.0)
        color = fr.color or found.get("color") or "NA"
        clarity = fr.clarity or found.get("clarity") or "NA"
        attributable = shape != "NA" and float(weight or 0) > 0

        note = f"certificateNo={fr.certificateNo}"
        if not attributable:
            note += " | UNATTRIBUTABLE: no stone attributes supplied and the " \
                    "certificate did not match inventory — stored for analytics " \
                    "only, cannot train a price cell"
            log.warning("feedback for %s has no resolvable stone — untrainable",
                        fr.certificateNo)

        rec = FeedbackRecord(
            stone_id=fr.stoneId or fr.certificateNo,
            decision=(Decision.OVERRIDE.value if desk is not None else Decision.REJECT.value),
            suggested_discount=fr.aiDiscount, suggested_net=0.0,
            shape_full=shape, weight=float(weight or 0.0),
            color=color, clarity=clarity,
            reason_code=fr.reason, note=note,
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
            # ONE definition of trainable, in feedback/store.py, shared with the
            # database column. Two copies of this rule is how they drifted apart
            # in the first place: the API said trainable=true while the stored
            # record could never train anything.
            "trainable": is_trainable(rec),
            "stone_resolved": attributable,
            "note": ("stored as a training label" if (desk is not None and attributable)
                     else ("stored for ANALYTICS ONLY — send deskDiscount (the desk's "
                           "own price) to make this a training label"
                           if desk is None else
                           "stored for ANALYTICS ONLY — the stone could not be "
                           "identified; send shape/weight/color/clarity, or a "
                           "certificateNo that matches your inventory")),
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
