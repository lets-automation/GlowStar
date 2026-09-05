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

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import SETTINGS
from ..features.build import build_features, get_target, TARGET
from ..reference.normalize import is_white_grid_color
from ..market.anchor import MarketTables, calibrate_offsets, anchor_predictions, market_series
from ..market.index import MarketIndex
from ..market.bgm import assess as bgm_assess
from ..feedback.learning import build_corrections, correction_for, as_training_examples
from .baseline import HierarchicalMedianModel
from .gbm import QuantileGBM

# Module-level logger, deliberately named `log` rather than reaching for the
# `logging` module inside methods. Several methods here do a local
# `import logging`, which makes `logging` a LOCAL name for that whole function —
# so any earlier reference to it in the same function raises
# `UnboundLocalError: cannot access local variable 'logging'`. That is not
# hypothetical: it took down the nightly retrain on 2026-08-10. A distinct
# module-level name cannot be shadowed that way.
log = logging.getLogger(__name__)


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
    #
    # DEFAULT 0.0 — the market anchor is OFF, and this was measured, not assumed.
    # Swept out-of-time (3,279 held-out sales, point-in-time grid):
    #     lam 0.00 -> MAE 2.07, >=5pt  8.1%
    #     lam 0.25 -> MAE 2.41, >=5pt 10.5%
    #     lam 0.50 -> MAE 3.25, >=5pt 18.1%   <- the old default
    #     lam 1.00 -> MAE 5.32, >=5pt 43.7%
    # Monotonic: every point of market weight makes the price worse. The anchor was
    # costing ~1.2 MAE and MORE THAN DOUBLING the >=5pt tail the desk complains about.
    #
    # Why it hurts: the table is an ASKING level (dealers' list prices), and asking
    # sits ~6 pts shallower than where this desk actually sells (market median -46.0
    # vs realized -53.8 on the same stones). Blending it in drags every stone toward
    # a price that is too expensive to trade — the exact "your price is too high"
    # the desk wrote on 68% of its corrections.
    #
    # The anchor's real job — carrying the CURRENT PER-CELL LEVEL — is now done far
    # better by the grid-routed model (see `_fit_grid_model`), which uses the
    # client's own live price sheet instead of a third party's asking prices.
    # Keep the market as a REFERENCE column and a sanity flag, not an input.
    #
    # Raise this ONLY with fresh evidence: re-run the sweep against a LIVE-built
    # market table (market.live.LiveMarket), and only if a lower tail comes with it.
    anchor_lambda: float = 0.0
    coverage: float = SETTINGS.interval_coverage
    # Days; down-weights older sales. It is a crude LEVEL tracker — the model cannot
    # extrapolate time, so recency weighting is what keeps it near the current level.
    #
    # Retuned 2026-07 on the full history and DELIBERATELY LEFT AT 30. A short
    # half-life looks better on one split and worse on another, i.e. it tracks how
    # much drift happens to sit in that test window, not a real property:
    #     split 06-01:  7d -> 2.417   30d -> 2.647   (short wins)
    #     split 06-20:  7d -> 2.364   30d -> 2.049   (short loses)
    # Do not tune this on a single split.
    recency_half_life: float = 30.0
    # FORWARD-DRIFT CORRECTION horizon, in days. 0 = OFF.
    #
    # OFF is measured, not assumed. The correction estimates how stale the model
    # gets over `bias_inner_days` and shifts every price by that amount. It was
    # calibrated at 45 days, which matched the old deployment: a laptop whose
    # nightly job missed 43% of runs, so the served model really was weeks stale.
    #
    # Production now retrains EVERY NIGHT on an always-on server. The model is one
    # day old, so the correction compensates for six weeks of drift that never
    # happens — and it now pushes the wrong way: the engine runs 0.60 too DEEP and
    # the correction deepens it further.
    #
    # A/B on five rolling origins, one engine fit per origin, arms differing only
    # by this switch (n=4,471 held-out sales):
    #     ON   MAE 1.9938   +/-2 64.4%   bias -0.60
    #     OFF  MAE 1.8297   +/-2 69.1%   bias +0.40      <- -0.164 MAE, +4.7pt
    # OFF won at all five origins. An independent audit measured -0.206 on nine
    # different origins; same direction, same size.
    #
    # The MECHANISM is not wrong, its horizon is. Do NOT delete the code: if the
    # market turns and the model goes stale-shallow again, set this to a horizon
    # that matches production (7-10 days), key it per shape, and let the promotion
    # gate choose between the two nightly.
    bias_inner_days: int = 0
    min_segment_samples: int = SETTINGS.min_segment_samples
    # Per-segment asking->realized offset: own calibration for liquid segments,
    # shrunk to the global offset by sample size for thin ones.
    anchor_offset_min_n: int = 25
    anchor_offset_shrink_k: float = 40.0
    # MARKET-LED pricing: replace the model's prediction with the cut+4C-matched
    # market median outright (model only as a fallback for stones with no match).
    #
    # DEFAULT False, and it should stay False. This flag was the shipped default in
    # `price_file.price_and_report` and it was never scored — the promotion gate,
    # `glowstar.status` and every backtest construct a plain EngineConfig(), i.e.
    # market_led=False. So the number we measured was NOT the number the client got.
    # Measured on the SAME held-out stones:
    #     market_led=False  MAE 3.84, within5 0.72, bias +0.85, coverage 0.82
    #     market_led=True   MAE 7.48, within5 0.47, bias +6.10, coverage 0.56
    # It ships an ASKING price labelled as a sale price: +6.1 pts too shallow (too
    # expensive), and the stated 80% band only holds 56% of the time.
    #
    # Keep it only for ablation. If it is ever re-enabled, the gate MUST score the
    # same config that ships.
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


def _is_strong_fluor(fluorescence) -> bool:
    """Strong / Very Strong only — the tiers with too little data to price."""
    from ..reference.normalize import normalize_fluorescence
    return normalize_fluorescence(
        None if fluorescence is None else str(fluorescence)) in ("Strong", "Very Strong")


def _fluor_band(color: str) -> str:
    """Colour band for fluorescence pricing. Fluoro bites hardest in colourless
    (D-E), fades through F-H, and is near-neutral in I-M. Used by BOTH the cap
    fit and the cap application, so they cannot drift apart."""
    c = (str(color) or "")[:1].upper()
    if c in ("D", "E"):
        return "D-E"
    if c in ("F", "G"):
        return "F-G"
    if c == "H":
        return "H"
    return "I-M"


class PricingEngine:
    def __init__(self, config: EngineConfig | None = None, tables: MarketTables | None = None):
        self.cfg = config or EngineConfig()
        self.tables = tables
        self.gbm: QuantileGBM | None = None
        # Second, GRID-ROUTED model: same features PLUS the client's own point-in-time
        # Master-grid reading for the stone's cell. Fit ONLY on rows that have a cell,
        # and used ONLY for stones that have one; everything else uses `self.gbm`.
        # Two models rather than one-with-a-missing-flag because it is measurably
        # better at BOTH ends: on stones WITH a cell the grid model wins (1.83 vs
        # 2.28 MAE), on stones WITHOUT one it LOSES (5.75 vs 5.01) — and if the grid
        # feed ever dies, a single grid-dependent model degrades to 4.12 while the
        # plain model holds at ~2.4. Routing keeps the win and removes that cliff.
        self.gbm_grid: QuantileGBM | None = None
        self.grid_history = None
        # Forward-drift correction per shape family (see `_fit_bias_correction`).
        # Empty until fit, and an empty dict is a no-op — so an engine pickled before
        # this existed still loads and simply applies no correction.
        self._bias_correction: dict = {}
        self.fallback = HierarchicalMedianModel(self.cfg.min_segment_samples)
        self.index: MarketIndex | None = None
        self.corrections: dict[str, dict] = {}     # per-segment offsets from feedback
        self._feedback_overrides: dict[str, float] = {}  # exact client decisions by StoneId
        self._shape_counts: dict[str, int] = {}
        self._offset: dict = {}                     # per-segment asking->realized offsets
        self._q_lo: float = 0.0        # conformal residual quantiles (signed)
        self._q_hi: float = 0.0
        self._defer_shapes: set[str] = set()   # shapes the model loses to the median on
        self._fluor_caps: dict = {}            # client's own realized fluoro discount caps
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

        # DROP rows with no usable target, ONCE, before anything downstream sees
        # them. The client's feed occasionally returns a sold stone with a
        # missing FDiscount: on 2026-08-10 exactly ONE row out of 17,367 did,
        # and it aborted the whole nightly retrain with
        # `ValueError: Input y contains NaN` inside MarketIndex's Ridge.
        #
        # Fixing it only in MarketIndex would have moved the crash rather than
        # removed it — HistGradientBoosting tolerates NaN in FEATURES but not in
        # the TARGET, so the GBM was the next thing to fall over. Filter here so
        # every downstream consumer (index, GBM, grid model, conformal, guard)
        # is guaranteed a clean target.
        #
        # `errors="coerce"` matters: the feed sends the target as a string, so a
        # non-numeric value must become NaN and be dropped rather than raise.
        n_before = len(train)
        train = train[pd.to_numeric(train[TARGET], errors="coerce").notna()]
        dropped = n_before - len(train)
        if dropped:
            # WARNING, not silence: one row is a feed hiccup, hundreds is a
            # broken feed, and the difference must be visible in the job log.
            log.warning("fit: dropped %d/%d training rows with no usable %s",
                        dropped, n_before, TARGET)
        if train.empty:
            raise ValueError(f"no training rows with a usable {TARGET}")

        feedback_records = feedback_records or []
        self.set_feedback(feedback_records)

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

        # GRID-ROUTED model. The dominant error term is the CURRENT LEVEL of the
        # stone's price cell — correcting it (oracle) halves MAE and cuts the >=5pt
        # tail 13.1% -> 3.7%, while a global/weekly correction does nearly nothing.
        # A tree cannot supply that itself (it cannot extrapolate time; ~40% of
        # served stones fall past its training window). The client's grid is that
        # level, joined POINT-IN-TIME so a past sale never sees a later edit.
        self._fit_grid_model(full, weights)

        # Fluoro-penalty caps from the client's OWN realized sales, so the GBM
        # (which over-penalises Strong/Very-Strong fluoro on colourless goods)
        # never discounts fluorescence deeper than the client historically does.
        self._fluor_caps = self._compute_fluor_caps(full)
        # THE CAPS ARE INERT UNDER THE SHIPPED CONFIG, and that must be loud.
        # They feed `_model_fluor_penalty` -> `market_extra` -> anchor_predictions
        # as `lam * (mkt + off + extra)`. Serving runs anchor_lambda = 0.0, so the
        # penalty is computed and multiplied by zero. Fluorescence is STILL
        # priced — it is a categorical feature in the GBM — so this is not an
        # accuracy bug; what is dead is the CAP. Trap 1 is the most carefully
        # documented trap in CLAUDE.md and `status` prints these values, so the
        # next person to tune them in response to a fluoro complaint would
        # measure exactly zero change and lose a day working out why.
        if self._fluor_caps and self.cfg.anchor_lambda == 0.0:
            log.warning(
                "fluoro caps computed (%d segments) but anchor_lambda=0.0, so they "
                "affect NO shipped price. Fluorescence is priced by the GBM feature; "
                "tuning _compute_fluor_caps will change nothing until the anchor is on.",
                len(self._fluor_caps))

        # Confidence bands via rolling-origin (forward) conformal calibration:
        # pooled residuals of the SHIPPED pipeline predicting unseen future months.
        # This captures genuine forward dispersion, so the stated coverage is honest
        # — a within-train adjacent slice understates it.
        #
        # The grid columns are attached POINT-IN-TIME first: without them every fold
        # would calibrate on the non-grid model while production routes ~92% of
        # stones through the grid model, i.e. an interval for a different function.
        conf_train = train
        if self.gbm_grid is not None and "grid_discount" not in train.columns:
            from ..market.grid_history import attach_grid
            conf_train = attach_grid(train, self.grid_history)

        # ORDER MATTERS. Estimate the forward-drift correction FIRST, on an inner
        # out-of-time slice, while `_bias_correction` is still empty (so
        # `_fold_predict` measures raw drift, not a corrected residual). Only then
        # calibrate the conformal — which now runs through the corrected path, so the
        # interval stays centred on the number we publish.
        self._bias_correction = self._fit_bias_correction(conf_train)
        self._q_lo, self._q_hi = self._rolling_conformal(conf_train)

        # Competence guard: shapes where the model+anchor does NOT beat the
        # segment-median baseline on an inner out-of-time slice defer to that
        # baseline (measured, not a hardcoded shape list). This protects rare
        # fancy shapes the global GBM extrapolates badly (e.g. Sq.Emerald priced
        # worse than a naive median) without touching shapes the model wins on.
        # Pass the SAME grid-attached frame the conformal used: the guard benches the
        # shipped pipeline, which routes through the grid model.
        self._defer_shapes = self._competence_defer_shapes(conf_train)
        if self._defer_shapes:
            import logging
            logging.getLogger(__name__).info(
                "Competence guard: deferring shapes to segment-median baseline: %s",
                sorted(self._defer_shapes))
        return self

    def _fit_grid_model(self, full: pd.DataFrame, weights: np.ndarray) -> None:
        """Fit the grid-routed model on the rows that have a point-in-time cell.

        Best-effort: with no grid history (fresh deploy, or the feed is down) this
        leaves `gbm_grid = None` and every stone routes to the plain model — the
        engine is never blocked on the grid.
        """
        from ..market.grid_history import GridHistory, attach_grid
        try:
            self.grid_history = GridHistory.load()
        except Exception:
            log = __import__("logging").getLogger(__name__)
            log.exception("Grid history unreadable; pricing without the grid feature.")
            self.grid_history = None
        if self.grid_history is None:
            return
        g = attach_grid(full, self.grid_history)      # per-row OrderDate => no leakage
        has = g["grid_discount"].notna().to_numpy()
        if int(has.sum()) < 1000:
            import logging
            logging.getLogger(__name__).warning(
                "Only %d rows have a point-in-time grid cell — not fitting the grid "
                "model (needs >=1000).", int(has.sum()))
            self.gbm_grid = None
            return
        sub = g[has]
        self.gbm_grid = QuantileGBM(coverage=self.cfg.coverage).fit(
            build_features(sub, self._month_base, with_grid=True),
            get_target(sub), sample_weight=weights[has])
        import logging
        logging.getLogger(__name__).info(
            "Grid-routed model fit on %d/%d rows (%.1f%% have a cell).",
            int(has.sum()), len(g), 100.0 * has.mean())

    # Shape families for the drift correction. Rounds and fancies age differently:
    # measured out-of-time, rounds carry ~no drift (+0.01) while fancies run ~+2.1
    # pts shallow, so one global number under-corrects fancies and over-corrects
    # rounds.
    _FANCY_SHAPES = frozenset({"Oval", "Pear", "Marquise", "Heart", "Cushion",
                               "Radiant", "Emerald", "Sq. Emerald", "Princess"})

    @classmethod
    def _shape_family(cls, shapes: pd.Series) -> np.ndarray:
        s = shapes.astype("string").str.strip().str.title()
        return np.where(s == "Round", "Round",
                        np.where(s.isin(cls._FANCY_SHAPES), "Fancy", "Other"))

    def _fit_bias_correction(self, train: pd.DataFrame, inner_days: int = 45,
                             min_n: int = 60, shrink_k: float = 50.0) -> dict:
        """Estimate the model's FORWARD DRIFT on an inner out-of-time slice.

        The model is trained on the past and served on the future, so it prices at a
        slightly stale level — measured, ~+0.9 pts too SHALLOW overall (too
        expensive), concentrated in fancies (+2.1) while rounds are unbiased (+0.01).
        That is the desk's standing complaint, and it is a level error, not a stone
        error.

        It MUST be measured out-of-time. Estimating it in-train gives ~0 by
        construction — the model fits its own training data — which is exactly why a
        naive in-train correction failed to transfer (in-train said -0.23 where the
        true forward bias was +2.75). Fitting on train[:-45d] and measuring on the
        last 45d reproduces the staleness the model will actually have in production.

        Measured on held-out sales: MAE 2.642 -> 2.360, >=5pt 13.9% -> 10.9%, bias
        +0.92 -> -0.40. The oracle (cheating) ceiling is 2.353, so this recovers
        essentially all of the available gain.

        Thin families are shrunk toward the global estimate by sample size, so a
        sparse family cannot chase noise.
        """
        if not int(getattr(self.cfg, "bias_inner_days", 0)):
            return {}                      # OFF by default — see EngineConfig
        inner_days = int(self.cfg.bias_inner_days)
        if "OrderDate_dt" not in train.columns or self.gbm is None:
            return {}
        cut = train["OrderDate_dt"].max() - pd.Timedelta(days=inner_days)
        inner_tr, inner_val = train[train["OrderDate_dt"] < cut], train[train["OrderDate_dt"] >= cut]
        if len(inner_tr) < 1500 or len(inner_val) < 300:
            return {}                        # too little to judge; no correction
        w = 0.5 ** ((inner_tr["OrderDate_dt"].max() - inner_tr["OrderDate_dt"]).dt.days
                    .to_numpy().astype(float) / self.cfg.recency_half_life)
        # `_bias_correction` is still empty here, so `_fold_predict` is uncorrected —
        # we are measuring the raw drift, not a corrected residual.
        pred = self._fold_predict(inner_tr, inner_val, w)
        resid = pred - inner_val["FDiscount"].to_numpy()
        glob = float(np.median(resid))
        fams = self._shape_family(inner_val["Shape_full"])
        out: dict = {"__global__": glob}
        for f in ("Round", "Fancy", "Other"):
            sel = fams == f
            n = int(sel.sum())
            if n < min_n:
                continue
            k = n / (n + shrink_k)
            out[f] = k * float(np.median(resid[sel])) + (1.0 - k) * glob
        import logging
        logging.getLogger(__name__).info(
            "Forward-drift correction (inner %dd, n_val=%d): %s",
            inner_days, len(inner_val), {k: round(v, 2) for k, v in out.items()})
        return out

    def _bias_shift(self, df: pd.DataFrame) -> np.ndarray:
        """Per-stone forward-drift correction to SUBTRACT from the prediction."""
        cal = getattr(self, "_bias_correction", None)
        if not cal:
            return np.zeros(len(df))
        glob = cal.get("__global__", 0.0)
        fams = self._shape_family(df["Shape_full"])
        return np.array([cal.get(f, glob) for f in fams], dtype=float)

    def _raw_predict(self, df: pd.DataFrame) -> np.ndarray:
        """Point prediction, routing each stone to the grid model where it has a
        point-in-time cell and to the plain model otherwise."""
        assert self.gbm is not None
        base = self.gbm.predict(build_features(df, self._month_base))
        if self.gbm_grid is None or "grid_discount" not in df.columns:
            return base
        has = df["grid_discount"].notna().to_numpy()
        if not bool(has.any()):
            return base
        out = base.astype(float).copy()
        sub = df[has]
        out[has] = self.gbm_grid.predict(
            build_features(sub, self._month_base, with_grid=True))
        return out

    def _competence_defer_shapes(self, train: pd.DataFrame, inner_days: int = 60,
                                 min_val: int = 20, margin: float = 0.5) -> set[str]:
        """Return shapes where the SHIPPED pipeline loses to the median baseline.

        Fits a temporary model on the older part of `train` and evaluates the
        deployed pipeline vs the hierarchical median on the most-recent inner slice,
        per shape. A shape is deferred only if the model's MAE is worse than the
        baseline's by more than `margin` points (hysteresis, so noise near parity
        doesn't flip a shape).

        CRITICAL: it must judge the pipeline that actually SHIPS. It used to bench a
        bare non-grid model — so it measured a weaker predictor than production runs,
        and then benched the shapes that predictor happened to lose on. Measured
        against the desk's own returned quotes, that mistake was expensive: the
        deferred shapes (Marquise/Pear/Sq.Emerald) scored MAE 3.87 with 38.5% of them
        >=5pts out, while the model-priced stones scored 1.54 / 6.4%. The guard was
        sending our best-priced fancies to a baseline that is worse than the model.
        Same failure mode as Trap 5 — benchmark the shipped path, or don't benchmark.
        """
        if self.gbm is None or self.tables is None or "OrderDate_dt" not in train.columns:
            return set()
        cut = train["OrderDate_dt"].max() - pd.Timedelta(days=inner_days)
        inner_tr = train[train["OrderDate_dt"] < cut]
        inner_val = train[train["OrderDate_dt"] >= cut]
        if len(inner_tr) < 1000 or len(inner_val) < 150:
            return set()                          # too little to judge; stay conservative
        w = 0.5 ** ((inner_tr["OrderDate_dt"].max() - inner_tr["OrderDate_dt"]).dt.days
                    .to_numpy().astype(float) / self.cfg.recency_half_life)
        # Judge the SHIPPED pipeline (grid routing included), not a bare model.
        anchored = self._fold_predict(inner_tr, inner_val, w)
        base = HierarchicalMedianModel(self.cfg.min_segment_samples).fit(inner_tr).predict(inner_val)
        actual = inner_val["FDiscount"].to_numpy()
        defer: set[str] = set()
        for shape, idx in inner_val.groupby("Shape_full", observed=True).groups.items():
            sel = inner_val.index.get_indexer(idx)
            if len(sel) < min_val:
                continue
            m_mae = float(np.mean(np.abs(anchored[sel] - actual[sel])))
            b_mae = float(np.mean(np.abs(base[sel] - actual[sel])))
            if m_mae > b_mae + margin:
                defer.add(shape)
        return defer

    def _fold_predict(self, past: pd.DataFrame, block: pd.DataFrame,
                      w: np.ndarray) -> np.ndarray:
        """Predict `block` from `past` using THE SHIPPED PIPELINE, in miniature.

        The conformal band is a promise about the number we actually publish, so it
        must be calibrated on that number. This previously fit a bare
        HistGradientBoostingRegressor and blended it via `anchor_predictions` — no
        monotonic constraints, no grid routing, and it ignored `market_led`
        entirely. That calibrates the interval for a function we do not ship (the
        shipped path's stated 80% band was holding ~56%).
        """
        mono = [QuantileGBM.MONOTONIC.get(c, 0)
                for c in build_features(past.head(1), self._month_base).columns]

        def _mk(monotonic):
            return HistGradientBoostingRegressor(
                loss="quantile", quantile=0.5, learning_rate=0.06, max_iter=300,
                max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                categorical_features="from_dtype", monotonic_cst=monotonic,
                random_state=42)

        tmp = _mk(mono).fit(build_features(past, self._month_base),
                            get_target(past), sample_weight=w)
        raw = tmp.predict(build_features(block, self._month_base))

        # Mirror the grid ROUTING: a stone with a point-in-time cell is priced by
        # the grid model in production, so its residual must come from one too.
        if self.gbm_grid is not None and "grid_discount" in past.columns:
            p_has = past["grid_discount"].notna().to_numpy()
            b_has = block["grid_discount"].notna().to_numpy()
            if p_has.sum() >= 500 and b_has.any():
                gmono = [QuantileGBM.MONOTONIC.get(c, 0) for c in build_features(
                    past.head(1), self._month_base, with_grid=True).columns]
                gsub = past[p_has]
                tmpg = _mk(gmono).fit(
                    build_features(gsub, self._month_base, with_grid=True),
                    get_target(gsub), sample_weight=w[p_has])
                raw[b_has] = tmpg.predict(
                    build_features(block[b_has], self._month_base, with_grid=True))

        if self.cfg.market_led:
            mkt = market_series(block, self.tables, cut_only=True).to_numpy()
            has = ~np.isnan(mkt)
            raw = raw.astype(float).copy()
            raw[has] = mkt[has]
        else:
            cal = self._offset if self.cfg.apply_asking_offset else 0.0
            raw = anchor_predictions(raw, block, self.tables, cal,
                                     lam=self.cfg.anchor_lambda)
        # Same drift correction the serving path applies, so the conformal residuals
        # are centred on the number we actually publish.
        return raw + self._trend_shift(block, None) - self._bias_shift(block)

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
            final = self._fold_predict(past, block, w)
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

    def set_feedback(self, records: list[dict]) -> None:
        """Apply returned decisions without mutating the persisted model.

        A re-price of a reviewed stone honours its exact client override. New
        stones can use only the guarded, sufficiently specific segment signal.
        """
        self.corrections = build_corrections(records or [])
        self._feedback_overrides = {
            str(r.get("stone_id")): float(r["human_discount"])
            for r in (records or [])
            if (r.get("decision") == "override" and r.get("human_discount") is not None
                and r.get("stone_id"))
        }

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
        # Cap the model's fluoro penalty ONLY where it demonstrably over-penalises.
        # Fluorescence genuinely guts a colourless D-E stone (milky look) and the
        # GBM learns that correctly (-15..-20, matching the desk's own quotes), so
        # capping D-E Medium/Strong/V.Strong was WRONG - it flattened a real effect
        # and put those stones 5-10pts off the desk's price. The GBM only
        # over-extrapolates where fluoro barely matters: low colours (I-M) and the
        # Faint tier. `_compute_fluor_caps` therefore only emits caps for those.
        caps = getattr(self, "_fluor_caps", None)
        if caps:
            from ..reference.normalize import normalize_fluorescence
            grp = df["Color"].astype("string").str[0].str.upper().map(_fluor_band)
            fl2 = df["Fluorescence"].map(normalize_fluorescence)
            # A cap is a FLOOR ON DEPTH ("never discount fluoro deeper than the desk
            # does"), never a premium. `_compute_fluor_caps` clamps every cap to <=0
            # for that reason — see there. -1e3 for an uncapped cell = no-op.
            allow = np.array([caps.get((g, f), -1e3) * 1.15 for g, f in zip(grp, fl2)])
            allow = np.minimum(allow, 0.0)        # belt-and-braces: never force a premium
            pen = np.maximum(pen, allow)          # both <=0: don't discount past the cap
        pen[~is_fl.to_numpy()] = 0.0
        return pen

    def _compute_fluor_caps(self, train: pd.DataFrame) -> dict:
        """The desk's OWN realized fluoro discount per (colour-band, tier):
        median(FDiscount at that fluoro) - median(FDiscount at None).

        Only emitted for the cells where the GBM over-penalises: the low-colour
        band (I-M, where fluoro is near-neutral) and the Faint tier (any colour).
        D-E/F-G/H at Medium+ are deliberately NOT capped - the model's deep
        penalty there matches what the desk actually charges. Measured against
        the desk's 122 returned prices: capping only these cells gives 12 stones
        >=5pts off / mean 2.06, vs 15 / 2.18 when D-H was capped wholesale."""
        from ..reference.normalize import normalize_fluorescence
        t = pd.DataFrame({
            "grp": train["Color"].astype("string").str[0].str.upper().map(_fluor_band),
            "fl": train["Fluorescence"].map(normalize_fluorescence),
            "fd": pd.to_numeric(train["FDiscount"], errors="coerce"),
        }).dropna(subset=["fd"])
        caps: dict = {}
        for grp in ("D-E", "F-G", "H", "I-M"):
            base = t.loc[(t["grp"] == grp) & (t["fl"] == "None"), "fd"].median()
            if pd.isna(base):
                continue
            for fl in ("Faint", "Medium", "Strong", "Very Strong"):
                if not (grp == "I-M" or fl == "Faint"):
                    continue          # never cap a real Medium+ penalty on D-E/F-G/H
                sub = t.loc[(t["grp"] == grp) & (t["fl"] == fl), "fd"]
                if len(sub) >= 20:
                    # CLAMP TO <= 0. The raw difference of two POOLED medians is
                    # confounded (different size/clarity mix per fluoro tier — the
                    # exact reason this whole cap table must be treated with
                    # suspicion), and it comes out POSITIVE on thin cells: I-M/Faint
                    # measured +0.5. Applied as `maximum(pen, +0.5*1.15)` that does
                    # not cap anything — it FORCES a fluorescence PREMIUM, quoting a
                    # fluorescent stone ~0.6pt shallower (more expensive) than the
                    # model wanted, feeding the shallow bias the desk complains about.
                    # A cap floors how DEEP we may go; it must never push shallower.
                    caps[(grp, fl)] = min(0.0, float(sub.median() - base))
        return caps

    def _model_bgm_penalty(self, df: pd.DataFrame) -> np.ndarray:
        """The MODEL's own BGM (milky/brown) penalty per stone (<=0 = deeper):
        model(actual BGM) - model(clean). Applied to the BGM-BLIND market anchor so
        a milky/brown stone prices DEEPER than the clean market, not identically.

        Without this, market_led=True forward pricing (the client deliverable)
        overwrites the model's prediction with the clean market and DISCARDS BGM
        entirely — the exact bug this fixes. Mirrors `_model_fluor_penalty`."""
        has_m = "milky_ord" in df.columns
        has_b = "brown_ord" in df.columns
        if not has_m and not has_b:
            return np.zeros(len(df))
        m = pd.to_numeric(df["milky_ord"], errors="coerce") if has_m else pd.Series(np.nan, index=df.index)
        b = pd.to_numeric(df["brown_ord"], errors="coerce") if has_b else pd.Series(np.nan, index=df.index)
        is_bgm = ((m.fillna(0) > 0) | (b.fillna(0) > 0)).to_numpy()
        if not bool(is_bgm.any()):
            return np.zeros(len(df))
        df2 = df.copy()
        df2["milky_ord"] = 0.0
        df2["brown_ord"] = 0.0
        pen = (self.gbm.predict(build_features(df, self._month_base))
               - self.gbm.predict(build_features(df2, self._month_base)))
        pen[~is_bgm] = 0.0
        return pen

    def _anchored_point(self, df: pd.DataFrame) -> np.ndarray:
        assert self.gbm is not None and self.tables is not None
        raw = self._raw_predict(df)
        # The model's fluorescence AND BGM penalties, injected into the (fluoro/BGM-
        # blind) market anchor so neither is diluted or discarded by market pricing.
        extra = self._model_fluor_penalty(df) + self._model_bgm_penalty(df)
        if self.cfg.market_led:
            # Forward pricing: price TO the clean cut+4C-matched market where it
            # exists (lambda=1, no history offset); the model carries any stone
            # with no cut-matched market. BGM/fluoro re-applied via `extra`.
            mkt = market_series(df, self.tables, cut_only=True).to_numpy()
            out = raw.astype(float).copy()
            has = ~np.isnan(mkt)
            out[has] = mkt[has] + extra[has]
            return out
        cal = self._offset if self.cfg.apply_asking_offset else 0.0
        anchored = anchor_predictions(raw, df, self.tables, cal, lam=self.cfg.anchor_lambda,
                                      market_extra=extra)
        # Correct the model's forward drift (it prices at a slightly stale level).
        return anchored - self._bias_shift(df)

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
        # NO GRID CELL. Measured on production: these are ~3.5% of volume, carry
        # ~4x the error of a fresh-cell stone (7.10 MAE vs 1.71), are wrong by
        # >=5 points four times in ten — and the published 80% band holds only
        # 54.5% of the time for them, because the conformal quantiles are global.
        #
        # Two-thirds of them previously raised NO flag at all, so the desk saw a
        # confident price with a normal-looking range and nothing telling them to
        # look. A confidence interval is a promise; this one was true on average
        # and false exactly where the desk needed it.
        #
        # The flag does not fix the band — per-bucket conformal is the real fix
        # and needs measuring — but it stops the silence, which is the part that
        # costs credibility.
        if "grid_discount" in row.index and pd.isna(row.get("grid_discount")):
            flags.append("no_grid_cell")
        return flags

    # --- prediction (batch) ---
    def predict(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> list[PriceSuggestion]:
        """Price a batch of stones. `as_of` sets the pricing month for the
        market-trend projection (default: each stone's OrderDate month, else the
        training reference month). For live forward pricing pass today's date."""
        assert self.tables is not None
        # Attach the client's own grid reading for each stone's cell, so the
        # grid-routed model can be used. At SERVE time the current grid is a
        # legitimate input (the desk itself prices from it); a caller backtesting
        # past sales must attach it POINT-IN-TIME beforehand (attach_grid with each
        # row's OrderDate), which is why an existing column is never overwritten.
        if getattr(self, "grid_history", None) is not None and "grid_discount" not in df.columns:
            from ..market.grid_history import attach_grid
            df = attach_grid(df, self.grid_history,
                             asof=as_of or pd.Timestamp.now().normalize())
        anchored = self._anchored_point(df)
        trend = self._trend_shift(df, as_of)
        fb = self.fallback.predict_detailed(df)
        mkt = market_series(df, self.tables, cut_only=self.cfg.market_led)
        direction = self.index.as_dict()["direction"] if self.index is not None else "flat"
        # getattr default keeps models pickled before the competence guard existed
        # loadable (registry/gated path): an old engine simply defers no shapes.
        defer_shapes = getattr(self, "_defer_shapes", set())

        out: list[PriceSuggestion] = []
        for i, (_, row) in enumerate(df.iterrows()):
            route_flags = self._route(row)
            # Competence guard: a shape the model loses to the segment median on
            # (measured out-of-time at fit time) is priced by the baseline and
            # flagged for human review rather than auto-priced by the model.
            #
            # A GRID CELL OVERRIDES THE DEFERRAL. The guard judges a SHAPE on an
            # inner 60-day slice; the fallback it defers to is the hierarchical
            # median, which is shape/size/colour/clarity only and cannot see the
            # grid at all. So a stone carrying the desk's own current price for its
            # exact cell was having that thrown away in favour of a shape-blind
            # median. Measured on the rolling 7-day horizon, the stones this guard
            # deferred (70 Pear in one week, 7 Sq.Emerald in another — every one of
            # them WITH a cell):
            #     deferred to the baseline   MAE 3.77   bias +2.30   33.8% >=5pt
            #     their grid cell alone      MAE 1.81
            #     the same shapes when not deferred: Pear 1.99, Sq.Emerald 1.95
            # Deferring a shape with 1,765 training rows off one inner slice is
            # noise, not a competence gap — and it is CLAUDE.md Trap 5's third head
            # (a guard benching stones against something worse than the model).
            #
            # The flag STILL fires, so the desk is told to look. Only the routing
            # changes, and only when there is a real cell to route on.
            deferred = row["Shape_full"] in defer_shapes
            if deferred:
                route_flags = route_flags + ["segment_review"]
            has_cell = ("grid_discount" in row.index
                        and pd.notna(row.get("grid_discount")))
            use_fallback = (("fancy_color" in route_flags) or ("rare_shape" in route_flags)
                            or (deferred and not has_cell))

            # BGM-as-base: clean-base prediction, then explicit BGM deduction.
            bgm = bgm_assess(row.to_dict(), self.tables)
            if bgm.assumes_no_bgm:
                route_flags = route_flags + ["bgm_unassessed"]
            # MEDIUM/HEAVY tinge: the model applies only a modest, severity-blind
            # BGM adjustment (thin training data on severe milky/brown), so it can
            # UNDER-discount a heavily-tinged stone. Flag it for the client's own
            # judgement rather than trust the small auto-deduction.
            try:
                mo, bo = float(row.get("milky_ord")), float(row.get("brown_ord"))
            except (TypeError, ValueError):
                mo, bo = 0.0, 0.0
            if (mo == mo and mo >= 2) or (bo == bo and bo >= 2):
                route_flags = route_flags + ["bgm_review"]

            # STRONG/V.STRONG fluorescence on NEAR-COLOURLESS (D-H): same principle
            # as bgm_review — flag it rather than pretend we can price it.
            # Fluorescence genuinely guts a colourless stone and the desk quotes it
            # very deep, but the training data is far too thin to learn how deep
            # (~37 V.Strong stones in a held-out window), so the model systematically
            # UNDER-discounts: measured out-of-time bias +1.47 (Strong) and +2.21
            # (V.Strong) vs -0.06 for None — i.e. we quote them too expensive, the
            # exact complaint the desk raises. A real miss: a 1.02ct D/VVS1 V.Strong
            # radiant the desk priced at -68 while we said -53.5.
            if _is_strong_fluor(row.get("Fluorescence")) and _fluor_band(
                    str(row.get("Color", ""))) in ("D-E", "F-G", "H"):
                route_flags = route_flags + ["fluor_review"]

            # Online feedback correction for this segment (from human overrides).
            corr = correction_for(self.corrections, row["Shape_full"], row["Weight"],
                                  row["Color"], row["Clarity"], row.get("CPS")) \
                if self.corrections else 0.0
            exact_override = getattr(self, "_feedback_overrides", {}).get(str(row.get("StoneId", "")))

            if exact_override is not None:
                disc = exact_override
                lo_d, hi_d = disc - 0.5, disc + 0.5
                method = "feedback_override"
                shift_applied = 0.0
                corr = 0.0
            elif use_fallback:
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
