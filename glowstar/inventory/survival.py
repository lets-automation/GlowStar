"""The censored, left-truncated survival dataset — Workstream B, Phase A.

Everything downstream (velocity model, bifurcation, repricing, the four reports)
is built on the frame this module produces. If the frame is wrong every number
the desk reads is wrong, so the corrections are asserted, not assumed.

THE CLOCK
---------
`Ageing` IS the client's own days-to-sell clock and its origin is
`MarketSheetDate`. Verified on the live book, exactly — not approximately:

    Sold  : Ageing == (OrderDate     - MarketSheetDate)   100% of rows
    Stock : Ageing == (snapshot_date - MarketSheetDate)   100% of rows

`AvailableDays` is a DIFFERENT quantity (~38% / ~61% agreement) and
`CreatedDate` is not the clock either (it precedes MarketSheetDate on ~96% of
stock). Re-derive before trusting:

    python -m glowstar.inventory.survival --check

TWO BIASES, IN OPPOSITE DIRECTIONS — both corrected or neither
---------------------------------------------------------------
1. RIGHT-CENSORING (makes the book look too FAST). Unsold stock contributes
   nothing to a sold-only average, so the slowest goods vanish. Stock is carried
   as `event=0`: it has taken AT LEAST its current age. That is not the same
   claim as "never sells", and the difference is the whole point.

2. LEFT-TRUNCATION (makes it look too SLOW). Sale records begin at a fixed date,
   but stock still holds stones listed years earlier. Those are survivors whose
   contemporaries that entered AND SOLD before the window are absent from the
   records entirely. Survivors in, successes out. So only stones that ENTERED
   inside the observed sales window are kept.

Correcting one and not the other is worse than correcting neither, because it
looks rigorous. MOU 5.4 stated as code. `service/tradeability.py` implements the
same pair for the live FrontOffice field; this module is the general form it
grew into, and the two are held together by
`tests/test_survival.py::test_agrees_with_the_shipped_tradeability_table`.

THE CENSORING DATE IS READ FROM THE DATA, NOT FROM THE WALL CLOCK
------------------------------------------------------------------
A stock stone is censored at the moment we last OBSERVED it unsold — the
snapshot date — not at `today`. If `records.json` is three days stale, censoring
at `today` silently adds three phantom days of "still unsold" to every stone in
the book and reports the desk as slower than they are. `observation_asof()`
recovers the snapshot date from the identity above and the report carries it, so
a stale feed shows up as an old date rather than as a slower desk.

LEAKAGE — one field here is a trap, and it is not an obvious one
-----------------------------------------------------------------
`Discount` looks like the listing discount (it is populated on 100% of Stock)
and is the obvious covariate for "price aggressiveness drives speed". It is NOT
usable: on a Sold row it has been OVERWRITTEN with the realized price. Measured
against the daily snapshots, on stones seen as Stock on 2026-06-18 and Sold by
2026-08-27 (n=4,475):

    now, on the Sold row:    Discount == FDiscount   92.6%
    then, on the Stock row:  Discount == FDiscount   30.2%   (mean gap 1.41 pts)
    BasePriceDiscount changed between the two snapshots      0.09%

So `Discount` means "asking" in the censored arm and "closing" in the event arm.
A model given it learns "this number looks like a closing price, therefore it
sold" and scores beautifully on nothing at all — CLAUDE.md Trap 2 in a new
costume. `BasePriceDiscount` is stable across the same window and equals the
final discount only 2% of the time, so THAT is the listing-time price covariate.
Guarded by `test_discount_is_never_a_velocity_covariate`.

Run:  python -m glowstar.inventory.survival
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.loaders import FORBIDDEN_FEATURES
from ..market.segments import cut_tier, size_band

log = logging.getLogger(__name__)

# A duration outside this range is a data error, not a slow stone.
MAX_DURATION_DAYS = 1500

# Covariates the velocity model may see. Everything here is knowable ON THE DAY
# THE STONE WAS LISTED — that is the whole test for membership.
CATEGORICAL_COVARIATES: tuple[str, ...] = (
    "shape", "color", "clarity", "cut_tier", "fluorescence", "lab", "location",
)
NUMERIC_COVARIATES: tuple[str, ...] = (
    "weight", "size_band", "rap_ppc", "base_discount",
    "brown_ord", "milky_ord", "shade_ord", "green_ord",
    "grid_discount", "grid_age_days", "listed_month",
)
COVARIATES: tuple[str, ...] = CATEGORICAL_COVARIATES + NUMERIC_COVARIATES

# Columns the frame carries for JOINING and REPORTING, which are never fed to a
# model. `Status` is one of them and it is worth being explicit about why: it is
# a post-listing fact, but it is exactly `event` spelled as a word, so it adds
# no information a model could cheat with — while the reports genuinely need to
# say "in stock" rather than "event=0". The split is enforced by
# `model_matrix()`, which only ever reads COVARIATES.
BOOKKEEPING_COLUMNS: tuple[str, ...] = (
    "StoneId", "Status", "duration", "event", "entered", "segment",
)

# Post-listing fields. `Discount` is in here for the reason in the docstring: it
# is the one that does not look like leakage. Market DEPTH is deliberately
# absent from COVARIATES too — see velocity.py; it is the second number, never a
# covariate, because merging it is exactly what MOU 5.2 forbids.
FORBIDDEN_VELOCITY_FEATURES: frozenset[str] = FORBIDDEN_FEATURES | frozenset({
    "Discount", "OrderDate", "OrderDate_dt", "Ageing", "AvailableDays",
    "Status", "IsDelivered", "IsRejected", "LeadStatus", "Ostatus",
    "duration", "event", "market_depth",
})

_UNASSESSED = -1.0        # loaders.py sentinel: NEVER NaN, NEVER 0.0 (0.0 = clean)


@dataclass
class SurvivalReport:
    """What the frame is, and what was thrown away to make it honest."""

    n_rows: int = 0
    n_events: int = 0                 # sold
    n_censored: int = 0               # still in stock
    window_start: pd.Timestamp | None = None
    observation_asof: pd.Timestamp | None = None
    n_dropped_left_truncated: int = 0
    n_dropped_bad_duration: int = 0
    oldest_truncated_days: int | None = None
    grid_cell_hit_rate: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        w = None if self.window_start is None else self.window_start.date()
        a = None if self.observation_asof is None else self.observation_asof.date()
        return (
            f"survival frame: {self.n_rows} rows "
            f"({self.n_events} sold / {self.n_censored} censored in stock) | "
            f"window {w} -> {a} | "
            f"left-truncation dropped {self.n_dropped_left_truncated} "
            f"(oldest {self.oldest_truncated_days}d before the window) | "
            f"bad durations dropped {self.n_dropped_bad_duration}"
        )


def observation_asof(df: pd.DataFrame) -> pd.Timestamp | None:
    """The date the STOCK arm was last observed unsold, recovered from the data.

    Uses the verified identity `Ageing == asof - MarketSheetDate` on Stock rows
    and takes the mode, so one malformed row cannot move it. Returns None when
    there is no stock to date the snapshot from; callers then fall back to today
    and the report says so.
    """
    stock = df[df["Status"] == "Stock"]
    if not len(stock):
        return None
    implied = (stock["MarketSheetDate_dt"]
               + pd.to_timedelta(pd.to_numeric(stock["Ageing"], errors="coerce"),
                                 unit="D"))
    implied = implied.dropna()
    if not len(implied):
        return None
    return pd.Timestamp(implied.dt.normalize().mode().iloc[0])


def _tinge(df: pd.DataFrame, col: str) -> np.ndarray:
    """Tinge ordinal with the UNASSESSED sentinel — never NaN, never 0.0.

    CLAUDE.md Trap 4: 0.0 means *assessed and clean*, and an all-NaN column
    hard-crashes HistGradientBoosting at fit, so a feed that stopped sending a
    field would take the nightly velocity retrain down with it.
    """
    if col not in df.columns:
        return np.full(len(df), _UNASSESSED)
    return pd.to_numeric(df[col], errors="coerce").fillna(_UNASSESSED).to_numpy(float)


def build_survival_frame(df: pd.DataFrame | None = None, *,
                         asof: pd.Timestamp | str | None = None,
                         grid: object | None = "auto",
                         max_duration: int = MAX_DURATION_DAYS
                         ) -> tuple[pd.DataFrame, SurvivalReport]:
    """The censored, left-truncation-guarded frame every velocity number rests on.

    Columns: `duration` (days), `event` (1 sold / 0 still in stock), `entered`
    (MarketSheetDate), the COVARIATES above, plus `StoneId` / `Status` /
    `segment` for joining and reporting. Never returns a frame with a
    post-listing field in the covariate set — that is asserted, not documented.

    `grid="auto"` loads the point-in-time grid history and joins each stone's
    cell AS OF ITS LISTING DATE. Pass `grid=None` to skip (the covariates then
    carry NaN, which HistGradientBoosting handles natively).
    """
    from ..data.loaders import load_records

    if df is None:
        df, _ = load_records()
    rep = SurvivalReport()

    for c in ("MarketSheetDate_dt", "OrderDate_dt"):
        if c not in df.columns:
            raise ValueError(f"{c} missing — load through data.loaders.load_records()")

    snap = observation_asof(df) if asof is None else pd.Timestamp(asof).normalize()
    if snap is None:
        snap = pd.Timestamp.now().normalize()
        rep.notes.append("no Stock rows to date the snapshot from — censored at today")
    rep.observation_asof = snap

    sold = df[df["Status"] == "Sold"].copy()
    sold["duration"] = (sold["OrderDate_dt"] - sold["MarketSheetDate_dt"]).dt.days
    sold["event"] = 1

    stock = df[df["Status"] == "Stock"].copy()
    stock["duration"] = (snap - stock["MarketSheetDate_dt"]).dt.days
    stock["event"] = 0                       # RIGHT-CENSORED, not "never sells"

    both = pd.concat([sold, stock], ignore_index=True)

    # LEFT-TRUNCATION GUARD. See the module docstring; this is MOU 5.4.
    window_start = sold["OrderDate_dt"].min()
    rep.window_start = window_start
    if pd.notna(window_start):
        keep = both["MarketSheetDate_dt"] >= window_start
        dropped = both[~keep]
        rep.n_dropped_left_truncated = int(len(dropped))
        if len(dropped):
            oldest = (window_start - dropped["MarketSheetDate_dt"].min()).days
            rep.oldest_truncated_days = int(oldest)
            log.info("Left-truncation guard: dropped %d stones listed before the "
                     "sales window opened (%s); oldest by %d days.",
                     len(dropped), window_start.date(), oldest)
        both = both[keep]

    good = both["duration"].between(0, max_duration)
    rep.n_dropped_bad_duration = int((~good).sum())
    both = both[good].copy()

    out = pd.DataFrame(index=both.index)
    out["StoneId"] = both["StoneId"].astype(str)
    out["Status"] = both["Status"].astype(str)
    out["duration"] = both["duration"].astype(float)
    out["event"] = both["event"].astype(int)
    out["entered"] = both["MarketSheetDate_dt"]

    # --- covariates, all knowable on the listing date ------------------------
    out["shape"] = both["Shape_full"].astype(str).str.strip().str.title()
    out["color"] = both["Color"].astype(str).str.strip().str.upper()
    out["clarity"] = both["Clarity"].astype(str).str.strip().str.upper()
    out["cut_tier"] = [cut_tier(c) for c in both["CPS"]]
    out["fluorescence"] = both["Fluorescence"].astype(str).str.strip().str.upper()
    out["lab"] = both["Lab"].astype(str).str.strip().str.upper()
    loc = both["Location"] if "Location" in both.columns else pd.Series("NA", index=both.index)
    out["location"] = loc.astype(str)
    out["weight"] = pd.to_numeric(both["Weight"], errors="coerce")
    out["size_band"] = [size_band(w) for w in out["weight"].fillna(0.0)]
    out["rap_ppc"] = pd.to_numeric(both["Rap"], errors="coerce")
    # BasePriceDiscount, NOT Discount — see the leakage section of the docstring.
    out["base_discount"] = pd.to_numeric(both["BasePriceDiscount"], errors="coerce")
    for c in ("brown_ord", "milky_ord", "shade_ord", "green_ord"):
        out[c] = _tinge(both, c)
    # Seasonality is NOT learnable on under a year of history (MOU 5.4). The
    # listing month is carried so the model can pick up an intake/liquidation
    # rhythm WITHIN the window; it is never presented as an annual seasonal
    # effect, and no report reads it as one.
    out["listed_month"] = out["entered"].dt.month.astype(float)

    out["segment"] = (out["shape"] + "|" + out["size_band"].astype(str) + "|"
                      + out["color"] + "|" + out["clarity"])

    # --- the grid cell as of LISTING (never as of sale) -----------------------
    if isinstance(grid, str) and grid == "auto":
        from ..market.grid_history import GridHistory
        grid = GridHistory.load()
        if grid is None:
            rep.notes.append("no grid history on disk — grid covariates are NaN")
    if grid is not None:
        from ..market.grid_history import attach_grid
        joined = attach_grid(both, grid, asof=None, date_col="MarketSheetDate_dt")
        out["grid_discount"] = pd.to_numeric(joined["grid_discount"],
                                             errors="coerce").to_numpy()
        out["grid_age_days"] = pd.to_numeric(joined["grid_age_days"],
                                             errors="coerce").to_numpy()
        rep.grid_cell_hit_rate = float(out["grid_discount"].notna().mean())
    else:
        out["grid_discount"] = np.nan
        out["grid_age_days"] = np.nan

    leaked = FORBIDDEN_VELOCITY_FEATURES & set(COVARIATES)
    if leaked:                    # belt and braces: the whitelist is the guard
        raise AssertionError(f"post-listing fields in the covariate set: {sorted(leaked)}")
    stray = set(out.columns) - set(COVARIATES) - set(BOOKKEEPING_COLUMNS)
    if stray:
        raise AssertionError(
            f"columns that are neither a covariate nor declared bookkeeping: "
            f"{sorted(stray)} — add them to one list or the other, so nobody has "
            f"to guess later whether a model may see them.")

    rep.n_rows = len(out)
    rep.n_events = int(out["event"].sum())
    rep.n_censored = int((out["event"] == 0).sum())
    log.info(rep.summary())
    return out.reset_index(drop=True), rep


# ---------------------------------------------------------------------------
# Kaplan-Meier with Greenwood intervals
# ---------------------------------------------------------------------------
@dataclass
class KMCurve:
    """A survival curve and the honest width around it."""

    times: np.ndarray
    survival: np.ndarray
    lower: np.ndarray                 # log-log transformed CI
    upper: np.ndarray
    n_at_risk: np.ndarray
    n_events: int
    n_censored: int
    alpha: float = 0.05

    def median(self) -> float:
        return self.quantile(0.5)

    def quantile(self, q: float) -> float:
        """First time the curve falls to <= 1-q. `inf` when never reached.

        `inf` is a RESULT, not a failure: within the window the segment simply
        does not turn over that far. Never substitute a point estimate for it —
        MOU 10.3, and `tradeability.py` has reported it this way since day one.
        """
        hit = np.nonzero(self.survival <= (1.0 - q))[0]
        return float(self.times[hit[0]]) if len(hit) else float("inf")

    def median_ci(self) -> tuple[float, float]:
        """Brookmeyer-Crowley interval for the median.

        The interval is the set of times whose CI for S(t) still straddles 0.5.
        The LOWER survival band sits below the curve, so it crosses 0.5 first
        and gives the EARLIEST plausible median; the UPPER band crosses last and
        gives the LATEST. (Getting these the wrong way round prints an interval
        that runs backwards — 35-33 days — which is how this was caught.) Either
        end is `inf` when that band never gets there: a wide answer is a correct
        answer on thin history (MOU 10.3).
        """
        lo_hit = np.nonzero(self.lower <= 0.5)[0]
        hi_hit = np.nonzero(self.upper <= 0.5)[0]
        lo = float(self.times[lo_hit[0]]) if len(lo_hit) else float("inf")
        hi = float(self.times[hi_hit[0]]) if len(hi_hit) else float("inf")
        return lo, hi

    def survival_at(self, t: float) -> float:
        """P(still unsold at day t)."""
        if not len(self.times) or t < self.times[0]:
            return 1.0
        return float(self.survival[np.searchsorted(self.times, t, side="right") - 1])


def km_curve(durations, observed, alpha: float = 0.05) -> KMCurve:
    """Kaplan-Meier estimator with Greenwood variance and log-log CIs.

    `observed` is True for a stone that actually sold, False for one still in
    stock. Log-log (rather than plain linear) intervals are used because they
    cannot stray outside [0, 1]: on thin segments, which is most of them here, a
    linear band routinely would, and a confidence bound of 1.4 on a probability
    destroys trust in every other number on the page.
    """
    from scipy.stats import norm

    d = np.asarray(durations, dtype=float)
    o = np.asarray(observed).astype(bool)
    if not len(d):
        e = np.array([], dtype=float)
        return KMCurve(e, e, e, e, e.astype(int), 0, 0, alpha)

    order = np.argsort(d, kind="mergesort")
    d, o = d[order], o[order]
    n = len(d)
    z = float(norm.ppf(1.0 - alpha / 2.0))

    times, surv, lows, highs, at_risk = [], [], [], [], []
    s, gw = 1.0, 0.0                  # survival, cumulative Greenwood sum
    i, n_left = 0, n
    while i < n:
        t = d[i]
        j = i
        events = 0
        while j < n and d[j] == t:
            events += int(o[j])
            j += 1
        if events and n_left > 0:
            s *= 1.0 - events / n_left
            if n_left > events:
                gw += events / (n_left * (n_left - events))
            else:
                gw = float("inf")   # everyone at risk sold at once: no information left
            times.append(float(t))
            surv.append(s)
            at_risk.append(n_left)
            if s <= 0.0 or s >= 1.0 or not np.isfinite(gw):
                lows.append(0.0 if s <= 0.0 else s)
                highs.append(1.0)
            else:
                # Var(log(-log S)) = greenwood / (log S)^2
                se = np.sqrt(gw) / abs(np.log(s))
                lows.append(float(s ** np.exp(z * se)))
                highs.append(float(s ** np.exp(-z * se)))
        n_left -= (j - i)
        i = j

    return KMCurve(
        times=np.asarray(times, dtype=float),
        survival=np.asarray(surv, dtype=float),
        lower=np.asarray(lows, dtype=float),
        upper=np.asarray(highs, dtype=float),
        n_at_risk=np.asarray(at_risk, dtype=int),
        n_events=int(o.sum()), n_censored=int((~o).sum()), alpha=alpha,
    )


def km_median(durations, observed) -> float:
    """Convenience: the KM median, or `inf` when it is not reached."""
    return km_curve(durations, observed).median()


# ---------------------------------------------------------------------------
def _check() -> None:
    """Re-derive the identities this module rests on. Cite the command, not me."""
    from ..data.loaders import load_records

    df, _ = load_records()
    d = df[df["Status"] == "Sold"]
    a = (d["OrderDate_dt"] - d["MarketSheetDate_dt"]).dt.days
    exact = (pd.to_numeric(d["Ageing"], errors="coerce") - a).abs().eq(0).mean()
    print(f"Sold : Ageing == OrderDate - MarketSheetDate      {exact:.4f} of rows")
    s = df[df["Status"] == "Stock"]
    snap = observation_asof(df)
    b = (snap - s["MarketSheetDate_dt"]).dt.days
    exact_s = (pd.to_numeric(s["Ageing"], errors="coerce") - b).abs().eq(0).mean()
    print(f"Stock: Ageing == {snap.date()} - MarketSheetDate   {exact_s:.4f} of rows")
    av = (pd.to_numeric(d["AvailableDays"], errors="coerce") - a).abs().eq(0).mean()
    print(f"       (AvailableDays would be {av:.4f} — a DIFFERENT quantity)")


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--check" in sys.argv:
        _check()
        return
    frame, rep = build_survival_frame()
    print(rep.summary())
    if rep.grid_cell_hit_rate is not None:
        print(f"grid cell at listing: {rep.grid_cell_hit_rate:.1%} of stones")
    for note in rep.notes:
        print(f"note: {note}")
    c = km_curve(frame["duration"], frame["event"].astype(bool))
    lo, hi = c.median_ci()
    print(f"\nwhole book: median {c.median():.0f}d  (95% CI {lo:.0f}-{hi:.0f}d) "
          f"from {c.n_events} sales + {c.n_censored} censored")
    naive = frame.loc[frame.event == 1, "duration"].median()
    print(f"  sold-only (censoring ignored) would say {naive:.0f}d — "
          f"the slow goods are invisible to it")


if __name__ == "__main__":
    main()
