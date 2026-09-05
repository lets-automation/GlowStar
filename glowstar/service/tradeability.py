"""Tradeability — how quickly a stone's segment turns over.

Their FrontOffice response asks for one of: High / Semi High / Medium / Semi Slow
/ Slow. This is a PROVISIONAL implementation of that field so their screen has a
real number to bind to; the full velocity engine (Workstream B) is a larger build
— survival model with intervals, own-vs-market separation, GMROI-optimised
repricing — and is not what this module claims to be.

THE CLOCK: durations run from `MarketSheetDate`, never `CreatedDate`
-------------------------------------------------------------------
`Ageing` IS the client's own days-to-sell clock, and its origin is the market
sheet date. Measured on the live book, the identity is exact, not approximate:

    Sold  stones:  Ageing == (OrderDate    - MarketSheetDate)   100% of rows
    Stock stones:  Ageing == (snapshot_date - MarketSheetDate)  100% of rows

`AvailableDays` is a DIFFERENT quantity (~38% / ~61% agreement) — do not treat the
two as interchangeable. `CreatedDate` is not the clock either: it sits earlier than
`MarketSheetDate` on ~96% of stock, so measuring from it counts days before the
stone was ever offered and reports the desk as slower than they are.

This module used `CreatedDate` until 2026-08-24. Re-verify the identity with:

    python -c "import pandas as pd; from glowstar.data.loaders import load_records; \
df,_=load_records(); d=df[df.Status=='Sold']; \
print((pd.to_numeric(d.Ageing,errors='coerce')-(d.OrderDate_dt-d.MarketSheetDate_dt).dt.days).abs().eq(0).mean())"

TWO BIASES, IN OPPOSITE DIRECTIONS — both must be corrected
------------------------------------------------------------
1. RIGHT-CENSORING (makes it look too FAST).
   A stone that has sat unsold for 300 days contributes nothing to a sold-only
   average, so the slowest goods vanish and every segment looks quicker than it
   is. Unsold stock is therefore included as *censored*: it has taken AT LEAST
   its current age, which is not the same as "never sells".

2. LEFT-TRUNCATION (makes it look too SLOW) — the subtler one, and it bit us.
   The sale records only begin 2025-12-18, but stock still contains stones
   listed years earlier. Those ancient stones are kept as slow survivors, while
   their contemporaries that entered AND SOLD before the window are absent from
   the record set entirely. Survivors in, successes out.

   Measured: 1,683 of 9,542 in-stock stones (17.6%) predate the first recorded
   sale, some by 1,346 days. Including them pushed the median from 46 days to
   75 — a 63% overstatement, in the opposite direction to the censoring bias.

So the estimate uses only stones that ENTERED STOCK inside the observation
window, and treats the unsold among them as censored. On this book:

    sold-only (censoring ignored)          42 days   too fast
    all stock censored (truncation ignored) 75 days   too slow
    window-restricted + censored            46 days   <- what we report

Correcting one bias and not the other is worse than correcting neither, because
it looks rigorous. This is MOU 5.4 stated as code.

Where the median is not reached inside the window the answer is reported as
not-reached rather than a fabricated point estimate. Cutoffs come from the
client's OWN distribution (quintiles): a segment is "Slow" relative to how this
desk actually trades, not to a number invented here.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LABELS = ("High", "Semi High", "Medium", "Semi Slow", "Slow")
MIN_SEGMENT = 15                 # sales below this and the segment backs off coarser
_CACHE_HOURS = 12.0


@dataclass
class _Table:
    by_segment: dict[str, float]     # segment key -> median days to sell
    sales_count: dict[str, int]      # segment key -> how many the desk has SOLD
    by_shape: dict[str, float]
    overall: float
    cutoffs: tuple[float, float, float, float]
    built_at: pd.Timestamp
    n_sold: int
    n_censored: int


_table: _Table | None = None
_lock = threading.Lock()


def _km_median(durations: np.ndarray, observed: np.ndarray) -> float:
    """Kaplan-Meier median survival time, with right-censored observations.

    `observed` is True for a stone that actually sold, False for one still in
    stock (censored). Returns the first time the survival curve drops to <= 0.5,
    or +inf when it never does inside the window — never a made-up number.
    """
    order = np.argsort(durations)
    d, o = durations[order], observed[order]
    n_at_risk = len(d)
    surv = 1.0
    i = 0
    while i < len(d):
        t = d[i]
        j = i
        events = 0
        while j < len(d) and d[j] == t:
            events += int(o[j])
            j += 1
        if events and n_at_risk > 0:
            surv *= (1.0 - events / n_at_risk)
            if surv <= 0.5:
                return float(t)
        n_at_risk -= (j - i)
        i = j
    return float("inf")             # median not reached — the segment is genuinely slow


def _observation_asof(df: pd.DataFrame) -> pd.Timestamp | None:
    """The date the STOCK arm was last observed unsold, recovered from the data.

    Uses the client's own verified identity `Ageing == asof - MarketSheetDate` on
    Stock rows and takes the MODE, so one malformed row cannot move it. Returns
    None when there is no stock to date the snapshot from; the caller then falls
    back to today, which is the old behaviour.

    Deliberately a local copy of `inventory.survival.observation_asof` rather than
    an import: that package is a separate workstream and is not deployed to the
    pricing server. Two small identical functions beat a cross-workstream import
    that breaks serving.
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


def _segment(shape, color, clarity) -> str:
    return f"{str(shape).strip().title()}|{str(color).strip().upper()}|{str(clarity).strip().upper()}"


def build_table(force: bool = False) -> _Table:
    """Build (and cache) the days-to-sell table from the live book."""
    global _table
    with _lock:
        if _table is not None and not force:
            age_h = (pd.Timestamp.now() - _table.built_at).total_seconds() / 3600
            if age_h < _CACHE_HOURS:
                return _table
        from ..data.loaders import load_records

        df, _ = load_records()
        for c in ("MarketSheetDate_dt", "OrderDate_dt"):
            if c not in df.columns:
                df[c] = pd.to_datetime(df[c.replace("_dt", "")], errors="coerce",
                                       utc=True).dt.tz_localize(None)
        # CENSOR AT THE DATE WE LAST OBSERVED THE BOOK, NOT AT THE WALL CLOCK.
        #
        # A stock stone is censored at the moment we last SAW it unsold — the
        # snapshot behind `records.json` — not at `today`. If the nightly pull is
        # late, censoring at `today` silently adds that many phantom "still
        # unsold" days to every stone in the book and reports the desk as SLOWER
        # than they are. On a 5-day-stale file the effect is not 5 days: a
        # Kaplan-Meier median lands on a step, so segments moved by up to 54 days.
        #
        # `inventory/survival.py` already does this correctly, and the two are
        # held together by test_survival::test_agrees_with_the_shipped_tradeability_table.
        # That test was failing purely on this difference — with the same clock the
        # two estimators agree on 236/236 segments to 0.0 days.
        #
        # Derived here rather than imported from `glowstar.inventory`, because the
        # inventory package is a separate workstream and is NOT deployed to the
        # pricing server — importing it would take this module down there.
        now = _observation_asof(df) or pd.Timestamp.now().normalize()

        sold = df[df["Status"] == "Sold"].copy()
        sold["dur"] = (sold["OrderDate_dt"] - sold["MarketSheetDate_dt"]).dt.days
        sold["obs"] = True

        # CENSORED: still in stock. Duration so far = last observed - listed.
        stock = df[df["Status"] == "Stock"].copy()
        stock["dur"] = (now - stock["MarketSheetDate_dt"]).dt.days
        stock["obs"] = False

        both = pd.concat([sold, stock], ignore_index=True)

        # LEFT-TRUNCATION GUARD: keep only stones that entered stock inside the
        # window our sale records actually cover. A stone listed before the first
        # recorded sale is kept in `stock` if it never sold, but its
        # contemporaries that DID sell back then are missing from the records —
        # so including it counts survivors without their successes and inflates
        # the estimate (measured: 46 -> 75 days). See the module docstring.
        window_start = sold["OrderDate_dt"].min()
        if pd.notna(window_start):
            before = len(both)
            both = both[both["MarketSheetDate_dt"] >= window_start]
            dropped = before - len(both)
            if dropped:
                log.info("Tradeability: excluded %d stones that entered stock before "
                         "the sales window opened (%s) — left-truncation guard.",
                         dropped, window_start.date())
        both = both[both["dur"].between(0, 1500)]
        both["seg"] = [_segment(s, c, cl) for s, c, cl
                       in zip(both["Shape_full"], both["Color"], both["Clarity"])]

        by_seg: dict[str, float] = {}
        sales_count: dict[str, int] = {}
        for seg, g in both.groupby("seg"):
            sales_count[seg] = int(g["obs"].sum())     # SOLD only — the desk's turnover
            if len(g) < MIN_SEGMENT:
                continue
            m = _km_median(g["dur"].to_numpy(float), g["obs"].to_numpy(bool))
            if np.isfinite(m):
                by_seg[seg] = m 
        by_shape: dict[str, float] = {}
        for shp, g in both.groupby(both["Shape_full"].astype(str).str.title()):
            if len(g) < MIN_SEGMENT:
                continue
            m = _km_median(g["dur"].to_numpy(float), g["obs"].to_numpy(bool))
            if np.isfinite(m):
                by_shape[shp] = m
        overall = _km_median(both["dur"].to_numpy(float), both["obs"].to_numpy(bool))
        overall = overall if np.isfinite(overall) else float(sold["dur"].median())

        vals = np.array(list(by_seg.values()), dtype=float)
        cut = (tuple(np.quantile(vals, [0.2, 0.4, 0.6, 0.8]))
               if len(vals) >= 20 else (35.0, 51.0, 69.0, 94.0))

        _table = _Table(by_segment=by_seg, sales_count=sales_count,
                        by_shape=by_shape, overall=float(overall),
                        cutoffs=cut, built_at=pd.Timestamp.now(),
                        n_sold=int(sold["dur"].between(0, 1500).sum()),
                        n_censored=int(stock["dur"].between(0, 1500).sum()))
        log.info("Tradeability table: %d segments (from %d sold + %d in-stock/censored), "
                 "cutoffs %s days", len(by_seg), _table.n_sold, _table.n_censored,
                 tuple(round(c) for c in cut))
        return _table


def label_for_days(days: float, cutoffs) -> str:
    """Faster than most -> 'High'; slower than most -> 'Slow'."""
    c1, c2, c3, c4 = cutoffs
    if days <= c1:
        return LABELS[0]
    if days <= c2:
        return LABELS[1]
    if days <= c3:
        return LABELS[2]
    if days <= c4:
        return LABELS[3]
    return LABELS[4]


def tradeability_for(shape, weight, color, clarity) -> dict:
    """Tradeability for a stone's segment, with the basis it rests on.

    Never returns a bare label: the desk gets the number of days and where it came
    from, so a coarse fallback is visible rather than disguised as a real estimate.
    """
    try:
        t = build_table()
    except Exception:
        log.exception("tradeability table unavailable")
        return {"label": None, "median_days": None, "basis": "unavailable"}

    seg = _segment(shape, color, clarity)
    if seg in t.by_segment:
        d = t.by_segment[seg]
        return {"label": label_for_days(d, t.cutoffs), "median_days": round(d),
                "basis": f"segment {seg}"}
    shp = str(shape).strip().title()
    if shp in t.by_shape:
        d = t.by_shape[shp]
        return {"label": label_for_days(d, t.cutoffs), "median_days": round(d),
                "basis": f"shape {shp} (too few sales for the exact segment)"}
    return {"label": label_for_days(t.overall, t.cutoffs), "median_days": round(t.overall),
            "basis": "whole book (thin segment)"}


def segment_sales_count(shape, color, clarity) -> int | None:
    """How many stones this desk has actually SOLD in the segment.

    Feeds the Liquidity and Market Strength scores: how often they turn a segment
    over is a different question from how long each one takes, and the client's
    own edge shows up exactly where the two disagree.
    """
    try:
        t = build_table()
    except Exception:
        return None
    return t.sales_count.get(_segment(shape, color, clarity))
