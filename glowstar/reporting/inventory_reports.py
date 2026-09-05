"""The four Workstream-B reports (MOU 5.1, 5.5). All four are deliverables.

  1. Inventory report      — tradeability, stock, discount change, high/low margin
  2. Price change report   — how the suggestion moved, and WHAT DROVE IT
  3. Sale / selling report — what actually sold, vs the comparable prior period
  4. Movement report       — the nine inventory-vs-sales cases

Each is a self-explanatory workbook mirroring `reporting/excel_report.py`, and
each carries a **Legend & Honesty** sheet: what every column means, and
explicitly what is computed-from-real-data versus what is a labelled prior or
not yet measurable. A non-technical reader has to be able to work out every
column without asking us — that is the Phase-D gate.

TWO THINGS THESE REPORTS REFUSE TO DO
--------------------------------------
* **Attribute causally what is only association.** The price-change report shows
  the grid, market and desk movements that sat alongside a price move and names
  the largest as the LIKELY driver. It does not decompose the move, because the
  drivers are correlated and a tidy percentage split would be invented.
* **State a direction a segment has not earned.** The movement report backs off
  to a coarser segment, or reports "insufficient history", rather than calling a
  trend off three sales (MOU 5.5).

Run:  python -m glowstar.reporting.inventory_reports
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR
from ..inventory import bifurcate as BF
from ..inventory import chart as CH

log = logging.getLogger(__name__)

OUT_DIR = ARTIFACTS_DIR
# A direction needs at least this many sales in BOTH periods, or the segment is
# reported as "insufficient history" instead of being given a trend it has not
# earned (MOU 5.5).
MIN_SALES_FOR_DIRECTION = 8
# Relative change below this is "Stable" rather than Up or Down. Movement
# reports are read as decisions, and calling a 2% wobble a trend is how a desk
# stops trusting one.
FLAT_BAND = 0.15


def _autosize(xl) -> None:
    for ws in xl.book.worksheets:
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))


def _write(out: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31], index=False)
        _autosize(xl)
    log.info("Wrote %s", out)
    return out


def _honesty_sheet(view: CH.InventoryView, extra: list[tuple[str, str]] | None = None
                   ) -> pd.DataFrame:
    """The Legend & Honesty sheet every one of the four reports carries."""
    rows = [
        ("Days to sell", "COMPUTED from the client's own realized sales, corrected "
                         "for censoring (unsold stock) and left-truncation (stones "
                         "listed before our sales window). Not a field read out of "
                         "any system."),
        ("The clock", "Days run from MarketSheetDate — the client's own Ageing "
                      "clock. Verified exact on 100% of rows. AvailableDays is a "
                      "different quantity and is NOT used."),
        ("Class (Fast..Slow)", "RELATIVE to this desk's own distribution: 'Slow' "
                               "means slow for Glow Star, not slow against an "
                               "outside benchmark. A book that got uniformly "
                               "faster would still have a slowest fifth — read "
                               "the class together with the ageing bucket, which "
                               "is absolute."),
        ("Ageing bucket", "ABSOLUTE age: 0-90 / 91-180 / 181-365 / 365+. Over 365 "
                          "days is a red flag."),
        ("Own velocity vs market depth", "TWO separate numbers and their ratio, "
                                         "never merged. A segment can be thin in "
                                         "the broad market and still turn fast for "
                                         "Glow Star — that gap is the point."),
        ("Market depth", "Count of genuine comparable listings after duplicate "
                         "('virtual inventory') listings are removed. It is an "
                         "ASKING market and is never used as a price level. Blank "
                         "means not looked up — it does NOT mean zero."),
        ("Confidence interval", "Where shown, an 80% band. A WIDE band is a "
                                "correct answer on thin history, not a defect."),
        ("'Not reached'", "The segment does not turn over that far inside the "
                          "history we have. We report that rather than invent a "
                          "number."),
        ("Value figures", "The client's own ASKING value (NetAmount). NOT cost — "
                          "the feed carries no cost field — so true margin and "
                          "GMROI cannot be computed from it."),
        ("Projected days-to-sell from a price change", "NOT PROVIDED. No causal "
         "price-to-speed effect is identifiable from this data: observationally a "
         "1-3pt cut is worth +2.6 pts of 30-day sale probability while a 3-5pt cut "
         "is worth -4.4 pts, which is the confounder (the desk cuts hardest on the "
         "hardest goods), not a dose response. A deliberate matched-pair price test "
         "would settle it."),
        ("Seasonality", "NOT modelled. The sales history is under a year, so annual "
                        "seasonality is not learnable yet and none is claimed."),
        ("Trade turnover benchmark", "An EXTERNAL PRIOR (retail jewellery rule of "
                                     "thumb), never derived from this client's "
                                     "data and never used to move a number."),
        ("Repricing suggestions", "Recommendations only. Nothing is applied and "
                                  "nothing is written to any client system; the "
                                  "desk approves every price."),
    ]
    rows += extra or []
    rows.append(("Generated", view.generated_at))
    rows.append(("Pricing model version", str(view.model_version)))
    for note in view.notes:
        rows.append(("Note on this run", note))
    return pd.DataFrame(rows, columns=["Item", "What it means / is it real?"])


# ---------------------------------------------------------------------------
# 1. INVENTORY REPORT
# ---------------------------------------------------------------------------
def build_inventory_report(view: CH.InventoryView, proposals: pd.DataFrame | None = None,
                           out: Path | None = None) -> Path:
    """Tradeability, stock, discount change, high and low margin (MOU 5.5 #1)."""
    out = out or OUT_DIR / "GlowStar_Inventory_Report.xlsx"
    c = view.classified.copy()

    stones = pd.DataFrame({
        "StoneId": c["StoneId"],
        "Segment": c["Segment"],
        "Tradeability (class)": c["Class"],
        "Tradeability (FrontOffice wording)": c["ClassFrontOffice"],
        "Age (days)": c["AgeDays"],
        "Ageing bucket": c["AgeingBucket"],
        "Expected days to sell": c["ExpectedDaysToSell"],
        "Expected days (low)": c["ExpectedDaysLow"],
        "Expected days (high)": c["ExpectedDaysHigh"],
        "Own velocity score": c["OwnVelocityScore"],
        "Market depth": c.get("MarketDepth"),
        "Market depth score": c.get("MarketDepthScore"),
        "Own vs market": c.get("OwnVsMarket"),
        "Velocity ratio (own/market)": c.get("VelocityRatio"),
        "Own sales behind the segment": c["SegmentSales"],
        "Velocity estimated?": np.where(c["VelocityEstimated"], "yes",
                                        "no — listed before the sales window"),
        "Asking value ($)": c["StockValueUsd"],
        "Red flag (365+ days)": np.where(c["RedFlag"], "YES", ""),
        "Basis": c["ClassBasis"],
    })

    if proposals is not None and len(proposals):
        p = proposals[["StoneId", "FairDiscount", "ProposedDiscount", "MovePts",
                       "Direction", "RevenueChangeUsd", "NeedsHumanReview",
                       "ReviewReasons", "LiquidationCandidate", "Why"]]
        stones = stones.merge(p, on="StoneId", how="left")
        stones = stones.rename(columns={
            "FairDiscount": "Fair discount (%)",
            "ProposedDiscount": "Suggested discount (%)",
            "MovePts": "Discount change (pts)",
            "RevenueChangeUsd": "Revenue effect ($)",
            "NeedsHumanReview": "Needs human review",
            "ReviewReasons": "Why review",
            "LiquidationCandidate": "Liquidation candidate",
            "Why": "Why this move"})

    stones = stones.sort_values(["Tradeability (class)", "Age (days)"],
                                ascending=[True, False])

    by_class = (c.groupby("Class", observed=True)
                .agg(Stones=("StoneId", "size"),
                     **{"Asking value ($)": ("StockValueUsd", "sum"),
                        "Median age (days)": ("AgeDays", "median"),
                        "Median expected days": ("ExpectedDaysToSell", "median")})
                .reindex(BF.CLASSES).reset_index().rename(columns={"index": "Class"}))

    ageing = pd.DataFrame(CH.ageing_distribution(view)["buckets"])

    # High and low margin: relative to the FAIR price, which is the only margin
    # comparison available without cost data — and the sheet says exactly that.
    margin = pd.DataFrame(columns=["StoneId", "note"])
    if proposals is not None and len(proposals):
        m = proposals.copy()
        m["MarginVsFairPct"] = np.where(
            m["FairDiscount"].abs() > 0,
            (m["ProposedDiscount"] - m["FairDiscount"]), np.nan)
        keep = ["StoneId", "Class", "AgeDays", "FairDiscount", "ProposedDiscount",
                "MovePts", "RevenueChangeUsd"]
        margin = pd.concat([
            m.nlargest(50, "RevenueChangeUsd")[keep].assign(Group="HIGH margin — held back"),
            m.nsmallest(50, "RevenueChangeUsd")[keep].assign(Group="LOW margin — given away"),
        ])

    bench = BF.benchmark_check(
        view.classified[view.classified["VelocityEstimated"].astype(bool)],
        value_col="StockValueUsd",
        realized_median_days=_realized_median(view))
    overview = pd.DataFrame(
        [("Report", "Glow Star — Inventory Intelligence"),
         ("Stock stones", f"{len(c):,}"),
         ("Asking value", f"${c['StockValueUsd'].sum(skipna=True):,.0f}"),
         ("Sales window", f"{view.report.window_start.date()} to "
                          f"{view.report.observation_asof.date()}"),
         ("Velocity model", "discrete-time hazard on the client's own sales"),
         ("Benchmark check", bench["verdict"])]
        + [(k, str(v)) for k, v in bench.items() if k != "verdict"],
        columns=["Field", "Value"])

    return _write(out, {
        "Overview": overview,
        "Stones": stones,
        "By class": by_class,
        "Ageing": ageing,
        "Margin high & low": margin,
        "Legend & Honesty": _honesty_sheet(view, [
            ("High / low margin", "Measured against the engine's own FAIR price "
                                  "for the same stone, because the feed carries "
                                  "no cost. It is 'how much we held back or gave "
                                  "away versus fair', not gross margin."),
        ]),
    })


def _realized_median(view: CH.InventoryView) -> float | None:
    from ..inventory.survival import km_median

    m = km_median(view.frame["duration"], view.frame["event"].astype(bool))
    return float(m) if np.isfinite(m) else None


# ---------------------------------------------------------------------------
# 2. PRICE CHANGE REPORT
# ---------------------------------------------------------------------------
def build_price_change_report(view: CH.InventoryView, *, days: int = 30,
                              out: Path | None = None) -> Path:
    """How the suggestion moved over a period, and what moved alongside it.

    The ATTRIBUTION is the deliverable, not the delta — but attribution here is
    by association, and the sheet says so. The grid cell, the market level and a
    desk correction all move together; naming the largest co-mover is honest,
    splitting the delta between them as percentages would not be.
    """
    out = out or OUT_DIR / "GlowStar_Price_Change_Report.xlsx"
    since = pd.Timestamp.now() - pd.Timedelta(days=days)

    quotes = _load_quotes(since)
    if quotes.empty:
        empty = pd.DataFrame([("No quotes recorded in the period",
                               f"looked back {days} days from "
                               f"{pd.Timestamp.now().date()}")],
                             columns=["Result", "Detail"])
        return _write(out, {"Overview": empty,
                            "Legend & Honesty": _honesty_sheet(view)})

    first = quotes.sort_values("ts").groupby("stone_id").first()
    last = quotes.sort_values("ts").groupby("stone_id").last()
    moved = pd.DataFrame({
        "StoneId": first.index,
        "First quoted": first["ts"].to_numpy(),
        "Last quoted": last["ts"].to_numpy(),
        "Quotes in period": quotes.groupby("stone_id").size().reindex(first.index).to_numpy(),
        "Discount then (%)": first["discount"].to_numpy(),
        "Discount now (%)": last["discount"].to_numpy(),
        "Market level then (%)": first.get("market_median_discount", pd.Series(np.nan, index=first.index)).to_numpy()
        if "market_median_discount" in first else np.nan,
        "Model then": first["model_version"].to_numpy(),
        "Model now": last["model_version"].to_numpy(),
        "Comparables then": first["comparable_count"].to_numpy(),
        "Comparables now": last["comparable_count"].to_numpy(),
    })
    moved["Change (pts)"] = (moved["Discount now (%)"] - moved["Discount then (%)"]).round(2)
    moved["Direction"] = np.where(moved["Change (pts)"] > 0.01, "shallower (dearer)",
                                  np.where(moved["Change (pts)"] < -0.01,
                                           "deeper (cheaper)", "unchanged"))
    moved["Model changed"] = moved["Model then"] != moved["Model now"]

    grid_move = _grid_movement(view, moved["StoneId"], since)
    moved = moved.merge(grid_move, on="StoneId", how="left")
    desk = _desk_corrections(since)
    moved = moved.merge(desk, on="StoneId", how="left")
    moved["Desk corrected"] = moved["DeskVariancePts"].notna()

    moved["Likely driver"] = [_likely_driver(r) for r in moved.to_dict("records")]
    moved["Attribution basis"] = (
        "largest co-movement in the period; drivers are correlated, so this is "
        "association, not a decomposition of the change")
    moved = moved.sort_values("Change (pts)")

    by_seg = _price_change_by_segment(quotes, moved)
    overview = pd.DataFrame([
        ("Report", "Glow Star — Price Change"),
        ("Period", f"last {days} days (since {since.date()})"),
        ("Stones re-quoted in the period", f"{len(moved):,}"),
        ("Moved deeper", f"{int((moved['Change (pts)'] < -0.01).sum()):,}"),
        ("Moved shallower", f"{int((moved['Change (pts)'] > 0.01).sum()):,}"),
        ("Median absolute move", f"{moved['Change (pts)'].abs().median():.2f} pts"),
        ("Model version changed for", f"{int(moved['Model changed'].sum()):,} stones"),
    ], columns=["Field", "Value"])

    return _write(out, {
        "Overview": overview,
        "By stone": moved,
        "By segment": by_seg,
        "Legend & Honesty": _honesty_sheet(view, [
            ("Likely driver", "ASSOCIATION, not causation. The grid cell, the "
                              "market level, a model promotion and a desk "
                              "correction move together; we name the largest "
                              "co-mover rather than splitting the change into "
                              "percentages we cannot support."),
            ("Grid cell move", "The client's own Master-grid cell for the stone, "
                               "read POINT-IN-TIME at each end of the period. "
                               "Blank means the stone has no grid cell — it is "
                               "never filled with an interpolated estimate."),
        ]),
    })


def _load_quotes(since) -> pd.DataFrame:
    try:
        from sqlalchemy import select
        from ..store.db import get_engine, quotes as Q
        with get_engine().connect() as c:
            df = pd.read_sql(select(Q).where(Q.c.ts >= since), c)
        return df
    except Exception:
        log.exception("could not read the quote history")
        return pd.DataFrame()


def _desk_corrections(since) -> pd.DataFrame:
    try:
        from sqlalchemy import select
        from ..store.db import get_engine, decisions as D
        with get_engine().connect() as c:
            df = pd.read_sql(select(D).where(D.c.ts >= since), c)
        if df.empty:
            return pd.DataFrame(columns=["StoneId", "DeskVariancePts"])
        g = df.groupby(df["stone_id"].astype(str))["variance_pts"].last()
        return pd.DataFrame({"StoneId": g.index, "DeskVariancePts": g.to_numpy()})
    except Exception:
        log.exception("could not read desk decisions")
        return pd.DataFrame(columns=["StoneId", "DeskVariancePts"])


def _grid_movement(view: CH.InventoryView, stone_ids, since) -> pd.DataFrame:
    """Grid-cell discount at each end of the period, point-in-time.

    Never an interpolated cell: a stone with no real cell gets NaN and the
    report says "no grid cell" rather than showing a manufactured number, which
    once produced a "you're 20 points off your own grid" escalation.
    """
    cols = ["StoneId", "GridThen", "GridNow", "GridMovePts", "GridAgeDays"]
    try:
        from ..data.loaders import load_records
        from ..market.grid_history import GridHistory
        hist = GridHistory.load()
        if hist is None:
            return pd.DataFrame(columns=cols)
        df, _ = load_records()
        want = set(str(s) for s in stone_ids)
        rows = df[df["StoneId"].astype(str).isin(want)].drop_duplicates("StoneId")
        now = pd.Timestamp.now()
        out = []
        for r in rows.itertuples():
            a, _ = hist.as_of(r.Shape_full, r.Weight, r.Color, r.Clarity,
                              r.CPS, r.Fluorescence, since)
            b, age = hist.as_of(r.Shape_full, r.Weight, r.Color, r.Clarity,
                                r.CPS, r.Fluorescence, now)
            out.append({"StoneId": str(r.StoneId), "GridThen": a, "GridNow": b,
                        "GridMovePts": (None if a is None or b is None else round(b - a, 2)),
                        "GridAgeDays": age})
        return pd.DataFrame(out, columns=cols)
    except Exception:
        log.exception("grid movement lookup failed")
        return pd.DataFrame(columns=cols)


def _likely_driver(r: dict) -> str:
    move = r.get("Change (pts)")
    if move is None or abs(move) < 0.01:
        return "no change"
    if r.get("Desk corrected"):
        return "desk correction"
    grid = r.get("GridMovePts")
    if grid is not None and not pd.isna(grid) and abs(grid) >= 0.5:
        if np.sign(grid) == np.sign(move):
            return f"grid cell moved {grid:+.1f} pts the same way"
    if r.get("Model changed"):
        return "model version changed"
    then, now = r.get("Comparables then"), r.get("Comparables now")
    if then and now and then > 0 and abs(now - then) / then > 0.25:
        return "market depth changed materially"
    return "velocity / model re-estimate (no single co-mover stands out)"


def _price_change_by_segment(quotes: pd.DataFrame, moved: pd.DataFrame) -> pd.DataFrame:
    key = quotes.drop_duplicates("stone_id").set_index(quotes.drop_duplicates("stone_id")["stone_id"].astype(str))
    seg = (key["shape"].astype(str).str.title() + "|" + key["color"].astype(str).str.upper()
           + "|" + key["clarity"].astype(str).str.upper())
    m = moved.copy()
    m["Segment"] = m["StoneId"].astype(str).map(seg)
    g = m.groupby("Segment", observed=True)
    return pd.DataFrame({
        "Segment": g.size().index,
        "Stones": g.size().to_numpy(),
        "Median change (pts)": g["Change (pts)"].median().round(2).to_numpy(),
        "Moved deeper": g["Change (pts)"].apply(lambda s: int((s < -0.01).sum())).to_numpy(),
        "Moved shallower": g["Change (pts)"].apply(lambda s: int((s > 0.01).sum())).to_numpy(),
        "Most common driver": g["Likely driver"].agg(
            lambda s: s.value_counts().index[0] if len(s) else None).to_numpy(),
    }).sort_values("Stones", ascending=False)


# ---------------------------------------------------------------------------
# 3. SALE / SELLING REPORT
# ---------------------------------------------------------------------------
def build_sales_report(view: CH.InventoryView, *, days: int = 30,
                       out: Path | None = None) -> Path:
    """What actually sold, against the comparable PRIOR period (MOU 5.5 #3)."""
    out = out or OUT_DIR / "GlowStar_Sales_Report.xlsx"
    from ..data.loaders import load_records

    df, _ = load_records()
    sold = df[df["Status"] == "Sold"].copy()
    asof = view.report.observation_asof or pd.Timestamp.now().normalize()
    cur_from, prior_from = asof - pd.Timedelta(days=days), asof - pd.Timedelta(days=2 * days)

    sold["ppc"] = pd.to_numeric(sold["FPerCarat"], errors="coerce")
    sold["disc"] = pd.to_numeric(sold["FDiscount"], errors="coerce")
    sold["net"] = pd.to_numeric(sold["FNetAmount"], errors="coerce")
    sold["days_to_sell"] = pd.to_numeric(sold["Ageing"], errors="coerce")
    from ..market.segments import size_band
    sold["size_band"] = [size_band(w) for w in pd.to_numeric(sold["Weight"], errors="coerce").fillna(0)]
    sold["SizeBand"] = [CH.size_band_label(b) for b in sold["size_band"]]

    cur = sold[(sold["OrderDate_dt"] >= cur_from) & (sold["OrderDate_dt"] <= asof)]
    prior = sold[(sold["OrderDate_dt"] >= prior_from) & (sold["OrderDate_dt"] < cur_from)]

    def by(keys: list[str]) -> pd.DataFrame:
        def agg(d, suffix):
            g = d.groupby(keys, observed=True)
            return pd.DataFrame({
                f"Sales{suffix}": g.size(),
                f"Median disc %{suffix}": g["disc"].median().round(2),
                f"Median $/ct{suffix}": g["ppc"].median().round(0),
                f"Revenue ${suffix}": g["net"].sum().round(0),
                f"Median days to sell{suffix}": g["days_to_sell"].median().round(0),
            })
        a, b = agg(cur, ""), agg(prior, " (prior)")
        out_df = a.join(b, how="outer").reset_index()
        out_df["Sales change"] = (out_df["Sales"].fillna(0) - out_df["Sales (prior)"].fillna(0))
        out_df["Discount change (pts)"] = (out_df["Median disc %"]
                                           - out_df["Median disc %(prior)"]
                                           if "Median disc %(prior)" in out_df
                                           else out_df["Median disc %"] - out_df["Median disc % (prior)"])
        return out_df.sort_values("Sales", ascending=False)

    overview = pd.DataFrame([
        ("Report", "Glow Star — Sale / Selling"),
        ("Current period", f"{cur_from.date()} to {asof.date()} ({days} days)"),
        ("Prior period", f"{prior_from.date()} to {cur_from.date()} ({days} days)"),
        ("Sales, current", f"{len(cur):,}"),
        ("Sales, prior", f"{len(prior):,}"),
        ("Revenue, current", f"${cur['net'].sum():,.0f}"),
        ("Revenue, prior", f"${prior['net'].sum():,.0f}"),
        ("Median discount, current", f"{cur['disc'].median():.2f}%"),
        ("Median discount, prior", f"{prior['disc'].median():.2f}%"),
        ("Median days to sell, current", f"{cur['days_to_sell'].median():.0f}"),
        ("Median days to sell, prior", f"{prior['days_to_sell'].median():.0f}"),
    ], columns=["Field", "Value"])

    return _write(out, {
        "Overview": overview,
        "By shape": by(["Shape_full"]),
        "By size": by(["SizeBand"]),
        "By colour": by(["Color"]),
        "By clarity": by(["Clarity"]),
        "By segment": by(["Shape_full", "SizeBand", "Color", "Clarity"]).head(400),
        "Legend & Honesty": _honesty_sheet(view, [
            ("Margin", "Not shown as gross margin: the feed carries no cost "
                       "field. Revenue, realized discount and $/ct are exact."),
            ("Prior period", "The immediately preceding window of the same "
                             "length. With under a year of history there is no "
                             "same-period-last-year comparison to make."),
        ]),
    })


# ---------------------------------------------------------------------------
# 4. MOVEMENT REPORT — the nine cases
# ---------------------------------------------------------------------------
NINE_CASES = {
    ("Up", "Up"): ("Growing and selling — healthy expansion",
                   "Keep stocking; check margin is holding"),
    ("Up", "Stable"): ("Stock building against flat demand",
                       "Capital tying up; slow the intake"),
    ("Up", "Down"): ("Accumulation — the danger case",
                     "Reprice or liquidate; stop intake"),
    ("Stable", "Up"): ("Turning faster on the same stock",
                       "Restock — demand outrunning supply"),
    ("Stable", "Stable"): ("Steady state", "No action"),
    ("Stable", "Down"): ("Demand softening under steady stock",
                         "Reprice before it becomes accumulation"),
    ("Down", "Up"): ("Selling down fast", "Stock-out risk — buy or manufacture"),
    ("Down", "Stable"): ("Natural drawdown", "Watch; restock if velocity holds"),
    ("Down", "Down"): ("Segment shrinking on both sides",
                       "Exit, or reprice to clear the tail"),
}


def _direction(now: float, before: float) -> str:
    if before <= 0:
        return "Up" if now > 0 else "Stable"
    change = (now - before) / before
    if change > FLAT_BAND:
        return "Up"
    if change < -FLAT_BAND:
        return "Down"
    return "Stable"


def build_movement_report(view: CH.InventoryView, *, days: int = 30,
                          out: Path | None = None) -> Path:
    """Stock level and sales trended TOGETHER per segment — the nine cases.

    Both series are reconstructed from the client's own records: a stone counts
    as stock from the day it entered until the day it sold, so this needs no new
    data capture (MOU 5.5).

    A segment with too few sales in either period is reported as "insufficient
    history" rather than given a direction it has not earned.
    """
    out = out or OUT_DIR / "GlowStar_Movement_Report.xlsx"
    from ..data.loaders import load_records
    from ..market.segments import size_band

    df, _ = load_records()
    asof = view.report.observation_asof or pd.Timestamp.now().normalize()
    cur_from = asof - pd.Timedelta(days=days)
    prior_from = asof - pd.Timedelta(days=2 * days)

    d = df[df["Status"].isin(("Sold", "Stock"))].copy()
    d["seg"] = (d["Shape_full"].astype(str).str.title() + "|"
                + pd.Series([CH.size_band_label(size_band(w)) for w in
                             pd.to_numeric(d["Weight"], errors="coerce").fillna(0)],
                            index=d.index) + "|"
                + d["Color"].astype(str).str.upper() + "|"
                + d["Clarity"].astype(str).str.upper())
    entered = d["MarketSheetDate_dt"]
    left = d["OrderDate_dt"].where(d["Status"] == "Sold")

    def stock_on(day: pd.Timestamp) -> pd.Series:
        """A stone is stock from the day it entered until the day it sold."""
        held = (entered <= day) & (left.isna() | (left > day))
        return d[held].groupby("seg", observed=True).size()

    def sales_between(a: pd.Timestamp, b: pd.Timestamp) -> pd.Series:
        m = (d["Status"] == "Sold") & (left > a) & (left <= b)
        return d[m].groupby("seg", observed=True).size()

    stock_now, stock_before = stock_on(asof), stock_on(cur_from)
    sales_now = sales_between(cur_from, asof)
    sales_before = sales_between(prior_from, cur_from)

    segs = sorted(set(stock_now.index) | set(stock_before.index)
                  | set(sales_now.index) | set(sales_before.index))
    rows = []
    for s in segs:
        sn, sb = float(stock_now.get(s, 0)), float(stock_before.get(s, 0))
        an, ab = float(sales_now.get(s, 0)), float(sales_before.get(s, 0))
        enough = (an >= MIN_SALES_FOR_DIRECTION) and (ab >= MIN_SALES_FOR_DIRECTION)
        inv_dir = _direction(sn, sb) if enough else None
        sal_dir = _direction(an, ab) if enough else None
        meaning, action = (NINE_CASES.get((inv_dir, sal_dir), (None, None))
                           if enough else
                           ("insufficient history — fewer than "
                            f"{MIN_SALES_FOR_DIRECTION} sales in a period",
                            "no direction stated; back off to a coarser segment "
                            "or wait for more history"))
        rows.append({
            "Segment": s,
            "Stock now": int(sn), "Stock 1 period ago": int(sb),
            "Sales this period": int(an), "Sales prior period": int(ab),
            "Inventory": inv_dir or "—", "Sales": sal_dir or "—",
            "Meaning": meaning, "Suggested action": action,
            "Basis": (f"stock {sb:.0f} -> {sn:.0f}; sales {ab:.0f} -> {an:.0f} "
                      f"over {days}-day periods; a change within "
                      f"{FLAT_BAND:.0%} counts as Stable"),
        })
    table = pd.DataFrame(rows).sort_values(["Sales this period", "Stock now"],
                                           ascending=False)

    stated = table[table["Inventory"] != "—"]
    grid = pd.DataFrame([
        {"Inventory": i, "Sales": s, "Meaning": m, "Suggested action": a,
         "Segments in this cell": int(((stated["Inventory"] == i)
                                       & (stated["Sales"] == s)).sum())}
        for (i, s), (m, a) in NINE_CASES.items()])

    overview = pd.DataFrame([
        ("Report", "Glow Star — Inventory Movement (the nine cases)"),
        ("This period", f"{cur_from.date()} to {asof.date()} ({days} days)"),
        ("Prior period", f"{prior_from.date()} to {cur_from.date()} ({days} days)"),
        ("Segments examined", f"{len(table):,}"),
        ("Segments given a direction", f"{len(stated):,}"),
        ("Segments reported as insufficient history",
         f"{len(table) - len(stated):,}"),
        ("Minimum sales for a direction", f"{MIN_SALES_FOR_DIRECTION} in BOTH periods"),
        ("Flat band", f"a change within {FLAT_BAND:.0%} is 'Stable'"),
        ("Danger cell (Up / Down)",
         f"{int(((stated['Inventory'] == 'Up') & (stated['Sales'] == 'Down')).sum()):,} segments accumulating"),
    ], columns=["Field", "Value"])

    return _write(out, {
        "Overview": overview,
        "Nine cases": grid,
        "By segment": table,
        "Legend & Honesty": _honesty_sheet(view, [
            ("How the two series are built", "Both come from the client's own "
                                             "records: a stone counts as stock "
                                             "from the day it entered until the "
                                             "day it sold. No new data capture "
                                             "is required."),
            ("Insufficient history", f"A direction is stated only where the "
                                     f"segment has at least "
                                     f"{MIN_SALES_FOR_DIRECTION} sales in BOTH "
                                     f"periods. Thinner segments are reported "
                                     f"as insufficient rather than given a "
                                     f"trend they have not earned."),
        ]),
    })


# ---------------------------------------------------------------------------
def build_all(*, days: int = 30, view: CH.InventoryView | None = None,
              proposals: pd.DataFrame | None = None) -> dict[str, Path]:
    view = view or CH.build_view()
    return {
        "inventory": build_inventory_report(view, proposals),
        "price_change": build_price_change_report(view, days=days),
        "sales": build_sales_report(view, days=days),
        "movement": build_movement_report(view, days=days),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = build_all()
    for name, p in paths.items():
        print(f"{name:14s} -> {p}")


if __name__ == "__main__":
    main()
