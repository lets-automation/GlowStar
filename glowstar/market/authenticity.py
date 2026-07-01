"""Market-data authenticity & cleaning (client priority).

Authentic market data is expensive and hard to get, so two things matter: (1)
make the most of what we have by cleaning it rigorously, and (2) be honest about
quality. Raw asking-price feeds are contaminated — ~90% of the Uni bulk dump was
duplicate re-listings of the same certificate. Feeding that raw would distort
the market level.

This module is the single, reusable cleaning pipeline for any Uni pull (live
comparables panel or bulk aggregate):

  normalize -> dedupe by certificate -> drop stale -> trim outliers
            -> score source quality -> robust median -> authenticity report

It handles both Uni response shapes (the export-report `data[]` rows and the
bulk dump rows) so callers don't special-case formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# --- normalization across the two Uni shapes ------------------------------


def _parse_pct(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace("%", "").replace(",", "").strip())
    except ValueError:
        return None


def _parse_date(v) -> datetime | None:
    if not v:
        return None
    if isinstance(v, dict):                      # bulk: {"$date": "..."}
        v = v.get("$date")
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def normalize_uni_stone(raw: dict) -> dict:
    """Map a raw Uni row (either shape) to a flat, typed comparable."""
    price = raw.get("price") or {}
    lab = raw.get("lab")
    lab_name = lab.get("lab") if isinstance(lab, dict) else lab
    cert = str(raw.get("certificateNumber") or raw.get("certificate_number") or "").strip()
    uid = cert or str(raw.get("diamondID") or raw.get("stone_uni_id") or "")
    disc = raw.get("stone_discount")
    if disc is None:
        disc = price.get("listDiscount")
    fl = raw.get("fluorescence")
    fl_int = fl.get("intensity") if isinstance(fl, dict) else fl
    return {
        "uid": uid,
        "cert": cert,
        "discount": _parse_pct(disc),
        "shape": raw.get("shape"),
        "size": raw.get("size"),
        "color": raw.get("color"),
        "clarity": raw.get("clarity"),
        "cut": raw.get("cut") or raw.get("cutShortTitle"),   # cut grade (EX/VG/GD) — moves price a lot
        "lab": lab_name,
        "fluorescence": fl_int,
        "cert_date": _parse_date(raw.get("stone_cert_date")
                                 or (lab.get("reportDate") if isinstance(lab, dict) else None)),
        "is_bgm": (raw.get("is_bgm") or "No"),
        "milky": raw.get("milky") or raw.get("Milky"),
        "shade": raw.get("shade_name") or raw.get("Shade"),
        "has_video": bool(raw.get("video_url")),
        "has_cert": bool(cert),
    }


# --- source-quality scoring ------------------------------------------------

_LAB_SCORE = {"GIA": 1.0, "IGI": 0.7, "HRD": 0.7}


def source_quality(stone: dict) -> float:
    """0..1 trust score: lab tier, certificate present, media present.

    Used to weight comparables and to filter very-low-trust listings.
    """
    score = _LAB_SCORE.get(str(stone.get("lab")).upper(), 0.4)
    if stone.get("has_cert"):
        score = min(1.0, score + 0.1)
    if stone.get("has_video"):
        score = min(1.0, score + 0.05)
    return round(score, 3)


# --- cleaning report -------------------------------------------------------

@dataclass
class AuthenticityReport:
    n_in: int = 0
    n_after_dedupe: int = 0
    n_stale_dropped: int = 0
    n_outlier_trimmed: int = 0
    n_used: int = 0
    duplicate_rate: float = 0.0
    median_discount: float | None = None
    mean_source_quality: float | None = None
    asof: str = ""

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class CleanResult:
    stones: list[dict] = field(default_factory=list)
    discounts: list[float] = field(default_factory=list)
    report: AuthenticityReport = field(default_factory=AuthenticityReport)


def clean_market_stones(
    raw_stones: list[dict], *, asof: datetime | None = None,
    max_age_days: int = 120, iqr_k: float = 2.0, min_quality: float = 0.0,
) -> CleanResult:
    """Run the full authenticity pipeline over raw Uni rows."""
    asof = asof or datetime.now(timezone.utc).replace(tzinfo=None)
    rep = AuthenticityReport(n_in=len(raw_stones), asof=asof.date().isoformat())

    norm = [normalize_uni_stone(s) for s in raw_stones]

    # 1) Dedupe by certificate (virtual/double-listed stones). Keep highest
    #    source quality per certificate.
    best: dict[str, dict] = {}
    for s in norm:
        if not s["uid"]:
            best[id(s)] = s                      # no id -> keep (can't dedupe)
            continue
        q = source_quality(s)
        prev = best.get(s["uid"])
        if prev is None or q > source_quality(prev):
            best[s["uid"]] = s
    deduped = list(best.values())
    rep.n_after_dedupe = len(deduped)
    rep.duplicate_rate = round(1 - len(deduped) / max(1, len(norm)), 3)

    # 2) Drop stale listings (cert date older than max_age_days), drop no-discount.
    fresh = []
    for s in deduped:
        if s["discount"] is None:
            continue
        cd = s["cert_date"]
        if cd is not None and (asof - cd).days > max_age_days:
            rep.n_stale_dropped += 1
            continue
        if source_quality(s) < min_quality:
            continue
        fresh.append(s)

    # 3) Trim discount outliers (IQR) — urgent sales / data errors.
    discs = np.array([s["discount"] for s in fresh], dtype=float)
    if len(discs) >= 8:
        q1, q3 = np.percentile(discs, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        keep_mask = (discs >= lo) & (discs <= hi)
        rep.n_outlier_trimmed = int((~keep_mask).sum())
        fresh = [s for s, k in zip(fresh, keep_mask) if k]
        discs = discs[keep_mask]

    rep.n_used = len(fresh)
    if len(discs):
        rep.median_discount = round(float(np.median(discs)), 2)   # median, not mean
        rep.mean_source_quality = round(float(np.mean([source_quality(s) for s in fresh])), 3)
    return CleanResult(stones=fresh, discounts=discs.tolist(), report=rep)


def asking_to_transaction(asking_discount: float, offset: float) -> float:
    """Convert an asking discount to an expected realized discount.

    `offset` is the calibrated asking->realized gap (negative: clients realize
    deeper discounts than asking). Learned from the client's own sales vs the
    market median (see market.anchor.calibrate_offset).
    """
    return asking_discount + offset
