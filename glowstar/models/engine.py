"""The Pricing Engine: the orchestrator that turns a stone into a priced,
bounded, explained suggestion (brief Sections 7.1-7.8).

Pipeline per stone:
  1. Route: fancy/cape color or rare shape or thin segment -> hierarchical
     fallback (wide band, human review). Otherwise -> the model path.
  2. Model path: leakage-free quantile GBM point prediction (recency-weighted),
     re-centered by the calibrated MARKET ANCHOR toward current market level,
     and (when soft attributes are supplied) adjusted by the market-learned
     BGM/milky/shade discount.
  3. Interval: split-CONFORMAL calibration on a recent held-out slice so the
     stated coverage is honest (empirically ~= target), not asserted.
  4. Output: discount, price/ct, net, interval, comparables, method, flags.

Every number is computed here; the LLM layer only narrates these outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import SETTINGS
from ..features.build import build_features, get_target
from ..reference.normalize import is_white_grid_color
from ..market.anchor import MarketTables, calibrate_offsets, anchor_predictions, market_series
from ..market.index import MarketIndex
from ..market.bgm import assess as bgm_assess
from ..feedback.learning import build_corrections, correction_for, as_training_examples
from .baseline import HierarchicalMedianModel
from .gbm import QuantileGBM


@dataclass
class PriceSuggestion:
    stone_id: str
    suggested_discount: float
    suggested_ppc: float
    suggested_net: float
    ci_discount_low: float
    ci_discount_high: float
    ci_net_low: float
    ci_net_high: float
    comparable_count: int
    market_median_discount: float | None
    method: str                       # "model" | "model+anchor" | "fallback"
    flags: list[str] = field(default_factory=list)
    market_direction: str = "flat"    # softening / firming / flat (trend layer)
    trend_shift_pts: float = 0.0      # damped forward de-bias applied (disc pts)
    # BGM-as-base: clean base + explicit deduction (client request).
    bgm_state: str = "unassessed"     # clean / bgm / unassessed
    bgm_deduction_pts: float = 0.0
    assumes_no_bgm: bool = True
    feedback_correction_pts: float = 0.0   # online correction from human overrides


@dataclass
class EngineConfig:
    split_date: str = SETTINGS.backtest_split_date
    # Market-anchor blend weight: final = (1-lam)*model + lam*market.
    # CLIENT DECISION (reviewed the flow): price 50/50 — half their own history
    # (the model), half the live market. lam=0.50. (Inner-validation favoured a
    # lower 0.25 for matching PAST realized discounts; the client deliberately
    # weights the live market higher for more market-responsiveness — a valid
    # trade of backtest fit for tracking the current market.)
    anchor_lambda: float = 0.50
    coverage: float = SETTINGS.interval_coverage
    recency_half_life: float = 30.0      # days; down-weights older sales (inner-validation best)
    min_segment_samples: int = SETTINGS.min_segment_samples
    # Per-segment asking->realized offset: own calibration for liquid segments,
    # shrunk to the global offset by sample size for thin ones.
    anchor_offset_min_n: int = 25
    anchor_offset_shrink_k: float = 40.0
    # MARKET-LED forward pricing. Default off (the backtest predicts the client's
    # PAST realized, history-first). For pricing NEW stones to sell, set True: the
    # price tracks the clean, CUT+4C-matched current market (where it exists),
    # with the model only as a fallback for stones with no cut-matched market.
    # This is what the client actually does — they price to their live UNI view.
    market_led: bool = False
    # For FORWARD LIST pricing (a new stone the client will LIST/advertise), the
    # client wants the market ASKING level, not the deeper expected-realized close.
    # When False, the asking->realized offset is NOT applied, so the market anchor
    # stays at the asking/list level. Keep True for valuing past/realized prices.
    apply_asking_offset: bool = True
    # Explicit market-trend (directional) projection. OFF by default: the FIXED
    # market_month_index feature already carries the time trend into the GBM, and
    # an extra projected shift double-counts it (measured: it re-introduced a
    # large +bias and broke interval coverage, MAE 3.9->5.1). The index is still
    # fit for the market_direction narration; the flag stays for ablation only.
    use_trend: bool = False
    trend_damping: float = 0.5
    trend_cap_pts: float = 6.0


class PricingEngine:
    def __init__(self, config: EngineConfig | None = None, tables: MarketTables | None = None):
        self.cfg = config or EngineConfig()
        self.tables = tables
        self.gbm: QuantileGBM | None = None
        self.fallback = HierarchicalMedianModel(self.cfg.min_segment_samples)
        self.index: MarketIndex | None = None
        self.corrections: dict[str, dict] = {}     # per-segment offsets from feedback
        self._shape_counts: dict[str, int] = {}
        self._offset: dict = {}                     # per-segment asking->realized offsets
        self._q_lo: float = 0.0        # conformal residual quantiles (signed)
        self._q_hi: float = 0.0
        self._train_max_date: pd.Timestamp | None = None
        self._train_ref_month: pd.Period | None = None
        self._month_base: pd.Timestamp | None = None   # frozen market_month_index origin

    # --- training ---
    def fit(self, train: pd.DataFrame, feedback_records: list[dict] | None = None) -> "PricingEngine":
        """Fit the engine. If `feedback_records` are supplied, human decisions
        are folded in: OVERRIDE/ACCEPT rows become weighted training labels
        (durable learning) and OVERRIDEs build per-segment online corrections."""
        if self.tables is None:
            self.tables = MarketTables.load()
        feedback_records = feedback_records or []
        self.corrections = build_corrections(feedback_records)

        # Augment training with human-validated labels (gold OVERRIDE labels
        # up-weighted; ACCEPT confirmations at base weight).
        fb_x, _, fb_w = as_training_examples(feedback_records)
        if len(fb_x):
            full = pd.concat([train, fb_x], ignore_index=True)
            weight_mult = np.concatenate([np.ones(len(train)), fb_w])
        else:
            full, weight_mult = train, np.ones(len(train))

        self._train_max_date = full["OrderDate_dt"].max()
        # Freeze the market_month_index origin to the training epoch so train,
        # test, conformal, and serving all share one time scale.
        self._month_base = full["MarketSheetDate_dt"].min()
        self._shape_counts = train["Shape_full"].value_counts().to_dict()
        self.fallback.fit(full)
        self._offset = calibrate_offsets(full, self.tables,
                                         min_n=self.cfg.anchor_offset_min_n,
                                         shrink_k=self.cfg.anchor_offset_shrink_k)

        # Market-trend (directional) layer: quality-adjusted index on train.
        self._train_ref_month = self._train_max_date.to_period("M")
        self.index = MarketIndex().fit(full)

        # Deployed point/interval model: fit on all data, recency-weighted, with
        # feedback up-weighting folded in.
        age = (self._train_max_date - full["OrderDate_dt"]).dt.days.to_numpy().astype(float)
        weights = (0.5 ** (age / self.cfg.recency_half_life)) * weight_mult
        self.gbm = QuantileGBM(coverage=self.cfg.coverage).fit(
            build_features(full, self._month_base), get_target(full), sample_weight=weights)

        # Confidence bands via rolling-origin (forward) conformal calibration:
        # pooled residuals of the full anchored+trend pipeline predicting unseen
        # future months. This captures genuine forward dispersion, so the stated
        # coverage is honest — a within-train adjacent slice understates it.
        self._q_lo, self._q_hi = self._rolling_conformal(train)
        return self

    def _rolling_conformal(self, train: pd.DataFrame) -> tuple[float, float]:
        months = sorted(train["OrderDate_dt"].dt.to_period("M").unique())
        period = train["OrderDate_dt"].dt.to_period("M")
        resids: list[float] = []
        for m in months[2:]:                          # need >=2 months of history
            past = train[period < m]
            block = train[period == m]
            if len(past) < 1000 or len(block) < 40:
                continue
            w = 0.5 ** ((past["OrderDate_dt"].max() - past["OrderDate_dt"]).dt.days
                        .to_numpy().astype(float) / self.cfg.recency_half_life)
            tmp = HistGradientBoostingRegressor(
                loss="quantile", quantile=0.5, learning_rate=0.06, max_iter=300,
                max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                categorical_features="from_dtype", random_state=42)
            tmp.fit(build_features(past, self._month_base), get_target(past), sample_weight=w)
            raw = tmp.predict(build_features(block, self._month_base))
            anchored = anchor_predictions(raw, block, self.tables, self._offset,
                                          lam=self.cfg.anchor_lambda)
            final = anchored + self._trend_shift(block, None)
            resids.extend((block["FDiscount"].to_numpy() - final).tolist())
        lo_p = (1.0 - self.cfg.coverage) / 2.0
        if len(resids) < 100:                         # fallback: in-sample residual
            final = self._final_point(train)
            resids = (train["FDiscount"].to_numpy() - final).tolist()
        arr = np.asarray(resids)
        return float(np.quantile(arr, lo_p)), float(np.quantile(arr, 1.0 - lo_p))

    def set_corrections(self, table: dict[str, dict]) -> None:
        """Update online feedback corrections without a full refit."""
        self.corrections = table or {}

    # --- internal: model point prediction, anchored ---
    def _model_fluor_penalty(self, df: pd.DataFrame) -> np.ndarray:
        """The MODEL's own fluorescence penalty per stone (<=0 = deeper for fluoro):
        model(actual) - model(fluoro=None). Applied to the fluorescence-BLIND market
        anchor so the market doesn't dilute the fluoro penalty the model already
        learned. Self-calibrating — matches the model (and the client), whereas the
        broad market over-penalises fluorescence."""
        if "Fluorescence" not in df.columns:
            return np.zeros(len(df))
        fl = df["Fluorescence"].astype("string").str.strip().str.lower()
        is_fl = ~fl.isin(["non", "none", "nan", ""]) & fl.notna()
        if not bool(is_fl.any()):
            return np.zeros(len(df))
        df2 = df.copy()
        df2["Fluorescence"] = "Non"
        pen = (self.gbm.predict(build_features(df, self._month_base))
               - self.gbm.predict(build_features(df2, self._month_base)))
        pen[~is_fl.to_numpy()] = 0.0
        return pen

    def _anchored_point(self, df: pd.DataFrame) -> np.ndarray:
        assert self.gbm is not None and self.tables is not None
        raw = self.gbm.predict(build_features(df, self._month_base))
        fluor_pen = self._model_fluor_penalty(df)   # deepens the market anchor for fluoro
        if self.cfg.market_led:
            # Forward pricing: price TO the clean cut+4C-matched market where it
            # exists (lambda=1, no history offset); the model carries any stone
            # with no cut-matched market.
            mkt = market_series(df, self.tables, cut_only=True).to_numpy()
            out = raw.astype(float).copy()
            has = ~np.isnan(mkt)
            out[has] = mkt[has] + fluor_pen[has]
            return out
        cal = self._offset if self.cfg.apply_asking_offset else 0.0
        return anchor_predictions(raw, df, self.tables, cal, lam=self.cfg.anchor_lambda,
                                  market_extra=fluor_pen)

    def _trend_shift(self, df: pd.DataFrame, as_of: pd.Timestamp | None) -> np.ndarray:
        """Damped, capped forward de-bias from the market-trend index.

        Projects the market level from the training reference month to each
        stone's pricing month (`as_of`, else its OrderDate month, else the
        training reference). Damped because naive extrapolation overshoots a
        mean-reverting market; capped to bound far-future projection.
        """
        if not self.cfg.use_trend or self.index is None or self._train_ref_month is None:
            return np.zeros(len(df))
        if as_of is not None:
            months = [as_of.to_period("M")] * len(df)
        elif "OrderDate_dt" in df.columns:
            months = [d.to_period("M") if pd.notna(d) else self._train_ref_month
                      for d in df["OrderDate_dt"]]
        else:
            months = [self._train_ref_month] * len(df)
        cap = self.cfg.trend_cap_pts
        shift = np.array([self.cfg.trend_damping * self.index.drift(self._train_ref_month, m)
                          for m in months])
        return np.clip(shift, -cap, cap)

    def _final_point(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> np.ndarray:
        """Anchored point prediction plus the damped market-trend shift."""
        return self._anchored_point(df) + self._trend_shift(df, as_of)

    # --- routing ---
    def _route(self, row: pd.Series) -> list[str]:
        flags: list[str] = []
        if not is_white_grid_color(str(row["Color"])):
            flags.append("fancy_color")
        if self._shape_counts.get(row["Shape_full"], 0) < self.cfg.min_segment_samples:
            flags.append("rare_shape")
        return flags

    # --- prediction (batch) ---
    def predict(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> list[PriceSuggestion]:
        """Price a batch of stones. `as_of` sets the pricing month for the
        market-trend projection (default: each stone's OrderDate month, else the
        training reference month). For live forward pricing pass today's date."""
        assert self.tables is not None
        anchored = self._anchored_point(df)
        trend = self._trend_shift(df, as_of)
        fb = self.fallback.predict_detailed(df)
        mkt = market_series(df, self.tables, cut_only=self.cfg.market_led)
        direction = self.index.as_dict()["direction"] if self.index is not None else "flat"

        out: list[PriceSuggestion] = []
        for i, (_, row) in enumerate(df.iterrows()):
            route_flags = self._route(row)
            use_fallback = ("fancy_color" in route_flags) or ("rare_shape" in route_flags)

            # BGM-as-base: clean-base prediction, then explicit BGM deduction.
            bgm = bgm_assess(row.to_dict(), self.tables)
            if bgm.assumes_no_bgm:
                route_flags = route_flags + ["bgm_unassessed"]

            # Online feedback correction for this segment (from human overrides).
            corr = correction_for(self.corrections, row["Shape_full"], row["Weight"],
                                  row["Color"], row["Clarity"]) if self.corrections else 0.0

            if use_fallback:
                disc = fb[i].discount + bgm.deduction_pts + corr   # fallback skips trend
                half = max(fb[i].spread, 6.0)
                lo_d, hi_d = disc - half, disc + half
                method = "fallback"
                shift_applied = 0.0
            else:
                disc = float(anchored[i]) + float(trend[i]) + bgm.deduction_pts + corr
                lo_d, hi_d = disc + self._q_lo, disc + self._q_hi
                method = "model+anchor" if not np.isnan(mkt.iloc[i]) else "model"
                shift_applied = float(trend[i])

            comp_n = int(self.tables.segments.get(
                self._seg_name(row), {}).get("n", 0)) if not np.isnan(mkt.iloc[i]) else 0
            out.append(self._build(row, disc, lo_d, hi_d, comp_n,
                                   None if np.isnan(mkt.iloc[i]) else float(mkt.iloc[i]),
                                   method, route_flags, direction, shift_applied, bgm, corr))
        return out

    def _seg_name(self, row: pd.Series) -> str:
        from ..market.segments import cut_graded, is_cut_aware_key, segment_keys
        keys = segment_keys(row["Shape_full"], row["Weight"], row["Color"], row["Clarity"],
                            row.get("CPS"))
        if self.cfg.market_led and cut_graded(row["Shape_full"]):
            keys = [k for k in keys if is_cut_aware_key(k)]   # cut-matched levels (backoff chain)
        for key in keys:
            name = "|".join(map(str, key)) if key else "__global__"
            rec = self.tables.segments.get(name)
            if rec and rec["n"] >= 12:
                return name
        return "__global__"

    def _build(self, row, disc, lo_d, hi_d, comp_n, mkt_med, method, flags,
               direction="flat", trend_shift=0.0, bgm=None, corr=0.0) -> PriceSuggestion:
        rap, wt = float(row["Rap"]), float(row["Weight"])
        def net(d): return rap * (1 + d / 100.0) * wt
        def ppc(d): return rap * (1 + d / 100.0)
        if comp_n and comp_n < 8:
            flags = flags + ["thin_market"]
        if abs(hi_d - lo_d) > 14:
            flags = flags + ["low_confidence"]
        if net(disc) >= SETTINGS.high_value_usd:
            flags = flags + ["high_value"]
        return PriceSuggestion(
            stone_id=str(row.get("StoneId", "")),
            suggested_discount=round(disc, 2),
            suggested_ppc=round(ppc(disc), 2),
            suggested_net=round(net(disc), 2),
            ci_discount_low=round(lo_d, 2), ci_discount_high=round(hi_d, 2),
            ci_net_low=round(net(lo_d), 2), ci_net_high=round(net(hi_d), 2),
            comparable_count=comp_n,
            market_median_discount=None if mkt_med is None else round(mkt_med, 2),
            method=method,
            flags=sorted(set(flags)),
            market_direction=direction,
            trend_shift_pts=round(trend_shift, 2),
            bgm_state=bgm.state if bgm else "unassessed",
            bgm_deduction_pts=bgm.deduction_pts if bgm else 0.0,
            assumes_no_bgm=bgm.assumes_no_bgm if bgm else True,
            feedback_correction_pts=round(corr, 2),
        )
