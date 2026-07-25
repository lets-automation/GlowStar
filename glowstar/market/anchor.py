"""Market anchor: re-center pricing toward the CURRENT market level and apply
soft-attribute (BGM/milky/shade) discounts learned from the Uni feed.

Why this exists (proven empirically): a model trained on the client's past
sales carries the *past* discount level. In a moving market the realized level
drifts, so the model is biased on recent stones — and the bias concentrates in
liquid commercial goods (rounds). The Uni market snapshot is ~contemporaneous
with the present, so its per-segment median discount carries the *current*
level. Blending the model with a calibrated market anchor pulls the level back
to now; the soft-attribute deltas correct for quality signals the client's own
data doesn't record yet.

Reference mismatch handled honestly: Uni discounts are *asking* prices from a
different dealer; the client's FDiscount is *realized*. We learn the
asking->realized offset per the most recent training window and apply it, rather
than assuming the two are on the same scale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR
from .aggregate_bulk import _milky_severity, _shade_class
from .segments import cut_graded, is_cut_aware_key, segment_keys, size_tag
from ..reference.normalize import normalize_fluorescence

log = logging.getLogger(__name__)

# Fluorescence penalises near-colourless goods; it is ~neutral for lower colours.
_NEAR_COLORLESS = frozenset("DEFGH")


@dataclass
class MarketTables:
    """Loaded market artifacts: segment medians + soft-attribute deltas."""

    segments: dict[str, dict]
    bgm: dict
    fluor: dict = field(default_factory=dict)   # market fluorescence deltas

    @classmethod
    def load(cls, artifacts_dir: Path | None = None) -> "MarketTables":
        d = artifacts_dir or ARTIFACTS_DIR
        segments = json.loads((d / "market_segments.json").read_text(encoding="utf-8"))
        bgm = json.loads((d / "bgm_discounts.json").read_text(encoding="utf-8"))
        fp = d / "fluor_discounts.json"
        fluor = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
        return cls(segments=segments, bgm=bgm, fluor=fluor)

    # --- market fluorescence adjustment ---
    def fluor_delta(self, color, fluorescence) -> float:
        """How much DEEPER the market lists this fluorescence level vs None, for
        this colour group. The segment median is pooled across fluorescence (mostly
        None), so it under-reflects a fluorescent stone; this delta corrects the
        MARKET anchor for it. Near-colourless (D-H) is penalised; lower colours are
        ~neutral (data-driven — returns 0 if the market shows no penalty)."""
        fl = normalize_fluorescence(fluorescence)
        if fl in ("None", "Unknown", None):
            return 0.0
        c = str(color or "").strip().upper()[:1]
        group = "nc" if c in _NEAR_COLORLESS else "low"
        rec = self.fluor.get("by_group", {}).get(f"{group}|{fl}")
        return float(rec["delta_vs_none"]) if rec else 0.0

    # --- market level (with hierarchical backoff; cut-aware when cut given) ---
    def market_median(self, shape: str, weight: float, color: str, clarity: str,
                      cut=None, min_n: int = 12, cut_only: bool = False) -> float | None:
        keys = segment_keys(shape, weight, color, clarity, cut)
        # Require a cut-matched market only where cut is GRADED (rounds). Fancies
        # have no cut grade, so they price to the 4C market (the cut-blind levels).
        # Keep the whole CUT-AWARE backoff chain (5->4->3->2 tuples), not just the
        # most-specific 5-tuple, so a thin top segment backs off to a well-supported
        # cut-matched parent (e.g. Round|8|J|VS2|VG thin -> Round|8|J|VG) instead of
        # giving up and dropping to the history model.
        if cut_only and cut is not None and cut_graded(shape):
            keys = [k for k in keys if is_cut_aware_key(k)]
        tag = size_tag(weight, shape)     # ROUND -> client's price slot
        for key in keys:
            name = "|".join(map(str, key)) if key else "__global__"
            # Prefer the SIZE-LOCAL live sub-bucket (name#0.80); fall back to the
            # bracket-level banked segment (name) if the sub-bucket is absent/thin.
            for nm in (name + tag, name):
                rec = self.segments.get(nm)
                if rec and rec["n"] >= min_n and not np.isnan(rec["median_discount"]):
                    return float(rec["median_discount"])
        return None

    # --- soft-attribute delta (extra discount vs clean) ---
    def soft_delta(self, milky_raw: str | None, shade_raw: str | None) -> float:
        delta = 0.0
        m = _milky_severity(milky_raw)
        s = _shade_class(shade_raw)
        if m != "none":
            delta += self.bgm["by_milky"].get(m, {}).get("delta_vs_clean", 0.0)
        if s == "negative":
            delta += self.bgm["by_shade"].get("negative", {}).get("delta_vs_clean", 0.0)
        return float(delta)


def market_series(df: pd.DataFrame, tables: MarketTables,
                  cut_only: bool = False) -> pd.Series:
    """Per-stone market median discount (NaN where no segment has support).

    Passes the stone's cut so a cut-aware market segment is preferred when the
    table has one (VG stones match VG market, not the EX-dominated blend).
    `cut_only=True` returns NaN unless a CUT-matched segment exists, so the
    caller can fall back to the model instead of a cut-blind market.
    """
    vals = [
        tables.market_median(r.Shape_full, r.Weight, r.Color, r.Clarity,
                             getattr(r, "CPS", None), cut_only=cut_only)
        for r in df.itertuples()
    ]
    return pd.Series([np.nan if v is None else v for v in vals], index=df.index)


def calibrate_offset(train: pd.DataFrame, tables: MarketTables,
                     recent_days: int = 45) -> float:
    """Global median (FDiscount - market_median) over the most recent window.

    The asking->realized (and residual time) offset, pooled across all segments.
    Kept as the shrinkage target for `calibrate_offsets` and for reference.
    """
    cut = train["OrderDate_dt"].max() - pd.Timedelta(days=recent_days)
    recent = train[train["OrderDate_dt"] >= cut]
    mkt = market_series(recent, tables)
    diff = (recent["FDiscount"] - mkt).dropna()
    return float(diff.median()) if len(diff) else 0.0


def calibrate_offsets(train: pd.DataFrame, tables: MarketTables,
                      recent_days: int = 45, min_n: int = 25,
                      shrink_k: float = 40.0) -> dict:
    """Per-segment asking->realized offsets, each shrunk toward the global offset.

    The Uni anchor is an *asking* level from other dealers; the client's
    FDiscount is *realized*. That gap is not constant across segments — liquid
    rounds close tighter to market than thin fancies — so a single global offset
    injects a segment-dependent bias (measured: it concentrates in fancies). We
    learn a per-segment median gap over the recent window and shrink it toward
    the global by sample size, w = n/(n+k), so thin segments fall back to global
    gracefully instead of chasing noise.
    """
    import collections

    cut = train["OrderDate_dt"].max() - pd.Timedelta(days=recent_days)
    recent = train[train["OrderDate_dt"] >= cut].copy()
    mkt = market_series(recent, tables)
    recent = recent.assign(resid=(recent["FDiscount"] - mkt)).dropna(subset=["resid"])
    glob = float(recent["resid"].median()) if len(recent) else 0.0

    buckets: dict[str, list[float]] = collections.defaultdict(list)
    for r in recent.itertuples():
        for key in segment_keys(r.Shape_full, r.Weight, r.Color, r.Clarity):
            name = "|".join(map(str, key)) if key else "__global__"
            buckets[name].append(float(r.resid))
    medians = {name: float(np.median(v)) for name, v in buckets.items()}
    counts = {name: len(v) for name, v in buckets.items()}
    return {"global": glob, "medians": medians, "counts": counts,
            "min_n": int(min_n), "shrink_k": float(shrink_k)}


def offset_for(cal: dict, shape: str, weight: float, color: str, clarity: str) -> float:
    """Resolve a stone's offset via hierarchical backoff, shrunk toward global."""
    if not cal:
        return 0.0
    glob, mn, k = cal["global"], cal["min_n"], cal["shrink_k"]
    for key in segment_keys(shape, weight, color, clarity):
        name = "|".join(map(str, key)) if key else "__global__"
        n = cal["counts"].get(name, 0)
        if n >= mn:
            w = n / (n + k)
            return w * cal["medians"][name] + (1.0 - w) * glob
    return glob


def anchor_predictions(model_pred: np.ndarray, df: pd.DataFrame, tables: MarketTables,
                       cal: dict | float, lam: float = 0.35,
                       market_extra: np.ndarray | None = None) -> np.ndarray:
    """Blend model prediction with the calibrated market anchor (per-segment offset).

    final = (1-lam)*model + lam*(market_median + offset_segment), only where a
    market segment has support; otherwise the model prediction is unchanged.
    `cal` is the dict from `calibrate_offsets`; a plain float is also accepted as
    a constant global offset (backward compatible).
    """
    mkt = market_series(df, tables).to_numpy()
    out = model_pred.copy().astype(float)
    for i, r in enumerate(df.itertuples()):
        if np.isnan(mkt[i]):
            continue
        off = offset_for(cal, r.Shape_full, r.Weight, r.Color, r.Clarity) \
            if isinstance(cal, dict) else float(cal)
        # `market_extra` deepens the (fluorescence-BLIND) market anchor by the
        # MODEL's own fluorescence penalty, so the market doesn't dilute how the
        # model already values fluoro. Self-calibrating: it matches the model (and,
        # empirically, the client), unlike the broader market which over-penalises.
        extra = float(market_extra[i]) if market_extra is not None else 0.0
        out[i] = (1.0 - lam) * model_pred[i] + lam * (mkt[i] + off + extra)
    return out
