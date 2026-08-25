"""FrontOffice API — the contract Glow Star's CRM asked for (spec 29-07-2026).

Three things their document asks for:

  1. BULK STONE PRICING — a list of stones in, one row out per stone:
     StoneId, Certificate No, AI Discount, Reason, Tradeability, AI Score,
     Confidence Score.
  2. FEEDBACK — certificate no + the desk's reason + the AI discount.
  3. PRICING MASTER DISCOUNT AI — price a GRID CELL (a weight RANGE + colour +
     clarity + cps + fluorescence), not an individual stone.

DESIGN RULE FOR THIS FILE
-------------------------
Their request schema is much richer than anything the engine can currently learn
from (eye-clean, luster, bowtie, blacks/whites, opens, naturals...). Two separate
questions, and conflating them is how a system starts inventing numbers:

  * CAN WE ACCEPT IT?  Yes — accept every field, always. Rejecting unknown fields
    would break their integration for no benefit.
  * CAN WE PRICE ON IT? Only if that field also exists in the SALES HISTORY the
    model trained on. Verified against the live inventory API: brown/green/milky/
    shade and the measurement block are there; eyeClean, luster, bowtie, iGrade,
    the black/white/open/natural family and kapan/article/grade are NOT.

So unknown-but-accepted fields are stored, not silently fed to the model. As they
accumulate in the daily snapshots they become learnable, and at that point they
are added deliberately, with a measured before/after — not by accident.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Request models — mirroring their document field-for-field
# --------------------------------------------------------------------------
class InventoryItemInclusion(BaseModel):
    """Appearance/inclusion block. All optional: a stone the desk has not fully
    inspected must still get a price."""
    brown: str | None = None
    green: str | None = None
    milky: str | None = None
    shade: str | None = None
    sideBlack: str | None = None
    centerSideBlack: str | None = None
    centerBlack: str | None = None
    sideWhite: str | None = None
    centerSideWhite: str | None = None
    centerWhite: str | None = None
    openCrown: str | None = None
    openTable: str | None = None
    openPavilion: str | None = None
    openGirdle: str | None = None
    girdleCondition: str | None = None
    efoc: str | None = None
    efot: str | None = None
    efog: str | None = None
    efop: str | None = None
    culet: str | None = None
    hna: str | None = None
    eyeClean: str | None = None
    ktoS: str | None = None
    naturalOnTable: str | None = None
    naturalOnGirdle: str | None = None
    naturalOnCrown: str | None = None
    naturalOnPavillion: str | None = None
    flColor: str | None = None
    graining: str | None = None
    redSpot: str | None = None
    luster: str | None = None
    certiComment: str | None = None
    bowtie: str | None = None
    iGrade: str | None = None


class InventoryItemMeasurement(BaseModel):
    depth: float | None = None
    table: float | None = None
    length: float | None = None
    width: float | None = None
    height: float | None = None
    crownHeight: float | None = None
    crownAngle: float | None = None
    pavilionDepth: float | None = None
    pavilionAngle: float | None = None
    girdlePer: float | None = None
    minGirdle: str | None = None
    maxGirdle: str | None = None
    ratio: float | None = None
    mGrade: str | None = None


class FrontOfficeStone(BaseModel):
    """One stone as their FrontOffice sends it."""
    stoneId: str
    uniqueId: str | None = None
    kapan: str | None = None
    article: str | None = None
    grade: str | None = None
    shape: str
    weight: float = Field(gt=0)
    color: str
    clarity: str
    cps: str | None = None
    cut: str | None = None
    polish: str | None = None
    symmetry: str | None = None
    fluorescence: str | None = None
    lab: str | None = None
    comments: str | None = None
    inclusion: InventoryItemInclusion | None = None
    measurement: InventoryItemMeasurement | None = None
    marketSheetDate: str | None = None
    availableDays: int | None = None
    certificateNo: str | None = None
    days: int | None = None


class FrontOfficeReason(BaseModel):
    """Their spec #2: certificate number + the desk's reason + the AI discount.

    NOTE the gap flagged back to the client: their document does not include the
    desk's OWN price. A reason with no corrected price tells us the number was
    wrong but not what right looks like — there is no label to learn from. So
    `deskDiscount` (or `deskPpc`) is accepted here and is what actually makes the
    record trainable; without it the reason is stored as analytics only.
    """
    certificateNo: str
    reason: str
    aiDiscount: float
    deskDiscount: float | None = None
    deskPpc: float | None = None
    stoneId: str | None = None
    user: str | None = ""
    # THE STONE ITSELF. Optional so the CRM's existing calls keep working, but
    # without these (or a certificate we can resolve) the record is UNTRAINABLE:
    # corrections are learned per PRICE CELL, and a cell needs shape/weight/
    # colour/clarity. The first version hardcoded "NA"/0.0 here, so 14 real desk
    # corrections — each one carrying the desk's own price, exactly what we had
    # asked them for — were stored with no stone attached and could never teach
    # anything. Supplied values win; otherwise we look the stone up.
    shape: str | None = None
    weight: float | None = None
    color: str | None = None
    clarity: str | None = None


def resolve_stone(certificate_no: str | None, stone_id: str | None) -> dict | None:
    """Find a stone's attributes from the inventory by certificate or id.

    Returns None when the stone is not ours — an external stone is a legitimate
    case, and the caller must then record the record as unattributable rather
    than invent values for it.
    """
    # 1. OUR OWN QUOTES FIRST. We priced this stone moments ago, so its
    #    attributes are in the audit trail and are always current.
    #
    #    records.json alone was not enough and the reason is subtle: it is
    #    rebuilt ONCE A DAY at 02:35, while the desk prices and corrects stones
    #    the same day they enter inventory. Real corrections arrived at 16:13
    #    and 16:37 on stones that were not in that morning's snapshot, so every
    #    one of them failed to resolve and was stored untrainable. Checking the
    #    next morning showed the stones present, which made the lookup look
    #    correct when it was simply asking a source up to 24h behind.
    try:
        from sqlalchemy import select, desc
        from ..store.db import get_engine, quotes
        for col, val in ((quotes.c.certificate_no, certificate_no),
                         (quotes.c.stone_id, stone_id)):
            if not val:
                continue
            with get_engine().connect() as c:
                row = c.execute(
                    select(quotes).where(col == str(val))
                    .order_by(desc(quotes.c.ts)).limit(1)).mappings().first()
            if row and row["shape"] and float(row["weight"] or 0) > 0:
                return {"shape_full": str(row["shape"]),
                        "weight": float(row["weight"]),
                        "color": str(row["color"] or "NA"),
                        "clarity": str(row["clarity"] or "NA")}
    except Exception:
        log.exception("quote lookup failed for %s / %s", certificate_no, stone_id)

    # 2. Fall back to the inventory snapshot (covers a stone we never priced).
    try:
        from ..data.loaders import load_records
        df, _ = load_records()
        for col, val in (("CertificateNo", certificate_no), ("StoneId", stone_id)):
            if not val or col not in df.columns:
                continue
            hit = df[df[col].astype(str) == str(val)]
            if len(hit):
                r = hit.iloc[-1]
                return {"shape_full": str(r.get("Shape_full") or "NA"),
                        "weight": float(r.get("Weight") or 0.0),
                        "color": str(r.get("Color") or "NA"),
                        "clarity": str(r.get("Clarity") or "NA")}
    except Exception:
        log.exception("could not resolve stone %s / %s", certificate_no, stone_id)
    return None


class MasterDiscountRequest(BaseModel):
    """Their spec #3: price a GRID CELL, not a stone.

    This is the Master-grid use case — 'what discount should this cell carry?' —
    so the answer is the level for the whole cell, with the support behind it.
    """
    fromWeight: float = Field(gt=0)
    toWeight: float = Field(gt=0)
    color: str
    clarity: str
    cps: str | None = None
    floro: str | None = None
    shape: str = "Round"
    # their doc: "some fields may be added like ratio, depth, height, diameter,
    # girdleperc etc" — accepted now so adding them later is not a breaking change
    ratio: float | None = None
    depth: float | None = None
    height: float | None = None
    diameter: float | None = None
    girdlePerc: float | None = None


# --------------------------------------------------------------------------
# Mapping their schema onto the engine's
# --------------------------------------------------------------------------
# Fields their spec sends that the SALES HISTORY does not contain, so the model
# has never seen them and cannot price on them yet. Accepted and echoed back as
# `received_not_yet_priced` so the integration is honest on its face.
_NOT_YET_LEARNABLE = (
    "sideBlack", "centerSideBlack", "centerBlack", "sideWhite", "centerSideWhite",
    "centerWhite", "openCrown", "openTable", "openPavilion", "openGirdle",
    "girdleCondition", "efoc", "efot", "efog", "efop", "culet", "hna", "eyeClean",
    "ktoS", "naturalOnTable", "naturalOnGirdle", "naturalOnCrown",
    "naturalOnPavillion", "flColor", "graining", "redSpot", "luster", "bowtie",
    "iGrade", "kapan", "article", "grade",
)


def to_stone_in(fo: FrontOfficeStone):
    """FrontOffice stone -> the engine's StoneIn.

    `cps` is preferred when supplied; otherwise it is derived from cut/polish/
    symmetry using the same rule as the Excel path, because the model learned the
    cut effect from the client's own clean grade vocabulary (3EX/EX/VG/GD/FR) and
    an unseen combined code silently loses the cut signal.
    """
    from .pricing_service import StoneIn
    from ..reporting.price_file import _make_cps

    inc = fo.inclusion or InventoryItemInclusion()
    mea = fo.measurement or InventoryItemMeasurement()
    cps = (fo.cps or "").strip() or _make_cps(fo.cut, fo.polish, fo.symmetry)
    return StoneIn(
        StoneId=fo.stoneId, Shape_full=fo.shape, Weight=fo.weight,
        Color=fo.color, Clarity=fo.clarity, CPS=cps,
        Fluorescence=(fo.fluorescence or "Non"), Lab=(fo.lab or "GIA"),
        Brown=inc.brown, Milky=inc.milky, Shade=inc.shade, Green=inc.green,
        Length=mea.length, Width=mea.width, Depth=mea.depth,
    )


def _confidence_score(facts: dict) -> int:
    """Confidence as 0-100 (their spec wants a score, not High/Medium/Low).

    Built from what actually drives reliability — the width of the calibrated
    interval, how much comparable market support the stone had, and whether the
    engine raised a flag saying it prices this kind of stone badly.
    """
    lo, hi = facts.get("ci_discount_low"), facts.get("ci_discount_high")
    width = abs(hi - lo) if lo is not None and hi is not None else 20.0
    # 6 pts wide -> ~90; 20 pts wide -> ~30
    score = max(0.0, min(100.0, 110.0 - 4.0 * width))
    comps = facts.get("comparable_count") or 0
    if comps < 8:
        score -= 25
    elif comps < 25:
        score -= 10
    flags = set(facts.get("flags") or [])
    if flags & {"fluor_review", "bgm_review", "rare_shape", "fancy_color", "no_grid_cell"}:
        score -= 20                     # we have told the desk we price these badly
    if facts.get("method") == "fallback":
        score -= 20
    return int(max(0, min(100, round(score))))


def price_stones(stones: list[FrontOfficeStone], service) -> list[dict]:
    """Their spec #1 — one response row per stone, in their field names."""
    from ..feedback.store import VARIANCE_REASON_THRESHOLD_PTS
    from .tradeability import tradeability_for, segment_sales_count
    from . import ai_score

    # PRICE THE WHOLE BOOK IN ONE MODEL CALL. This is the endpoint the CRM uses
    # for 500+ stone runs; per-stone it ran ~95 ms/stone, so a 5,000-stone request
    # took ~8 minutes against nginx's 300 s timeout. Batched: ~2.3 ms/stone.
    # Answers are identical — verified stone-for-stone, not assumed.
    #
    # Stones that fail to CONVERT (bad weight, missing colour) are kept as
    # per-stone errors exactly as before: one malformed row must never cost the
    # desk the other 4,999 prices.
    prepared: list = [None] * len(stones)
    conv_err: dict[int, Exception] = {}
    for i, fo in enumerate(stones):
        try:
            prepared[i] = to_stone_in(fo)
        except Exception as e:
            conv_err[i] = e
    idx = [i for i in range(len(stones)) if i not in conv_err]
    priced = service.price_many([prepared[i] for i in idx]) if idx else []
    by_i: dict[int, object] = dict(zip(idx, priced))

    out: list[dict] = []
    for i, fo in enumerate(stones):
        row: dict[str, Any] = {"StoneId": fo.stoneId,
                               "CertificateNo": fo.certificateNo}
        try:
            if i in conv_err:
                raise conv_err[i]
            stone = prepared[i]
            res = by_i[i]
            if isinstance(res, Exception):
                raise res
            f = res["suggestion"]
            # Use the CANONICALISED shape (StoneIn normalises it), not the raw
            # `fo.shape`: the client sends codes like RBC/OB, and the segment
            # tables are keyed by the trained name. Passing the raw code here
            # silently found no segment and cost the stone its tradeability and
            # Liquidity/MarketStrength scores.
            shape = stone.Shape_full
            trade = tradeability_for(shape, fo.weight, fo.color, fo.clarity)
            conf = _confidence_score(f)
            # `days` / `availableDays` is how long THIS stone has been in stock —
            # the only per-stone input the Urgency score has.
            age = fo.days if fo.days is not None else fo.availableDays
            scores = ai_score.compute(
                our_discount=f["suggested_discount"],
                market_discount=f.get("market_median_discount"),
                market_depth=f.get("comparable_count"),
                own_sales=segment_sales_count(shape, fo.color, fo.clarity),
                median_days=trade["median_days"],
                age_days=(float(age) if age is not None else None),
                confidence=conf,
            )
            row.update({
                "AIDiscount": f["suggested_discount"],
                "AIPricePerCarat": f["suggested_ppc"],
                "AITotal": f["suggested_net"],
                "FairRangeLow": f["ci_discount_low"],
                "FairRangeHigh": f["ci_discount_high"],
                "Reason": _reason_text(res, f),
                "Tradeability": trade["label"],
                "TradeabilityDays": trade["median_days"],
                "TradeabilityBasis": trade["basis"],
                "NeedsReview": bool(set(f.get("flags") or []) &
                                    {"fluor_review", "bgm_review", "rare_shape",
                                     "fancy_color", "thin_market"}),
                "Flags": f.get("flags") or [],
                "ReasonRequiredAbovePts": VARIANCE_REASON_THRESHOLD_PTS,
                "Error": None,
                **scores,
                # their spec names the headline field "AI Score"
                "AIScore": scores["FinalAIScore"],
            })
            # Durable audit + the evidence needed to refit the score weights.
            # Best-effort by design: a store outage must never cost a price.
            _persist(fo, f, scores, trade, service)
            unused = _unused_fields(fo)
            if unused:
                row["ReceivedNotYetPriced"] = unused
        except Exception as e:                      # one bad stone never fails a book
            log.exception("FrontOffice pricing failed for %s", fo.stoneId)
            row.update({"AIDiscount": None, "Reason": None, "Tradeability": None,
                        "ConfidenceScore": 0, "AIScore": None, "FinalAIScore": None,
                        "NeedsReview": True,
                        "Error": f"{type(e).__name__}: {e}"})
        out.append(row)
    return out


def model_version(service) -> str | None:
    """Which model produced this price — the audit trail's whole point.

    Read from the registry rather than from the service object: the service holds
    a loaded engine, not its card, and a quote row that cannot name the model that
    made it is not an audit record. Cached on the service after the first lookup.
    """
    cached = getattr(service, "_model_version_cache", None)
    if cached is not None:
        return cached or None
    v = None
    try:
        from ..models import registry
        v = registry.current_version()
    except Exception:
        log.exception("could not resolve the live model version")
    try:
        service._model_version_cache = v or ""
    except Exception:
        pass                      # a read-only/mock service: just skip the cache
    return v


def _persist(fo, facts: dict, scores: dict, trade: dict, service) -> None:
    """Record the quote and its scores. Never raises — see store/db.py."""
    try:
        from ..store import db
        mv = model_version(service)
        db.record_quote(facts=facts, stone=fo.model_dump(), model_version=mv,
                        source="frontoffice")
        db.record_scores(stone_id=fo.stoneId, s=scores, tradeability=trade,
                         model_version=mv)
    except Exception:
        log.exception("could not persist quote/scores for %s", fo.stoneId)


def _reason_text(res: dict, facts: dict) -> str:
    """The plain-English 'why' their desk reads."""
    exp = res.get("explanation")
    if isinstance(exp, dict):
        for k in ("text", "summary", "explanation", "why"):
            if exp.get(k):
                return str(exp[k])
        return "; ".join(str(v) for v in exp.values() if isinstance(v, str))[:600]
    if exp:
        return str(exp)
    return (f"{abs(facts['suggested_discount']):.1f}% below Rapaport, from "
            f"{facts.get('comparable_count', 0)} comparable market stones "
            f"({facts.get('method')}).")


def _unused_fields(fo: FrontOfficeStone) -> list[str]:
    """Fields they sent that the model cannot price on yet — named, not hidden."""
    inc = fo.inclusion
    sent = []
    for name in _NOT_YET_LEARNABLE:
        v = getattr(inc, name, None) if inc is not None else None
        if v is None:
            v = getattr(fo, name, None)
        if v not in (None, "", "NA"):
            sent.append(name)
    return sent
