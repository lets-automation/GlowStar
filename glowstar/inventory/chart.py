"""Dashboard payloads — JSON endpoints, NOT a web screen (MOU 5.1, 7.6, 11.2).

MOU 7.6 rules the screen out in as many words: *the Angular application is the
chat interface for Workstream D only. It is not a dashboard or analytics UI for
Workstreams A-C, which remain out of scope under 11.2.* MOU 5.1 names the
deliverable "Dashboard data — JSON endpoints". So the endpoints ARE the product,
and a screen is new scope needing a 12 amendment.

That has a design consequence, and it is the whole reason this module is written
the way it is: **whoever eventually renders these — the client's own team, or us
under a later amendment — must need no business logic on their side.** Every
payload therefore carries its own basis, its own units, and its own limitations
in the JSON. Nothing is left for a caller to reconstruct, and no number appears
without the support behind it.

Two rules inherited by any future renderer:

  * Own velocity and market depth stay TWO numbers plus a ratio. There is no
    blended headline gauge here, however much better one would look on a
    dashboard — MOU 5.2 and 8.1, and the gap between the two is the client's
    edge.
  * Read-and-recommend only (MOU 11.3). Nothing here writes to any client
    system, and no price is applied — the desk approves in their own system.

Run:  python -m glowstar.inventory.chart
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..market.segments import SIZE_EDGES
from . import bifurcate as BF

log = logging.getLogger(__name__)

# Coarse quality tiers for the heatmap axis. Deliberately coarse: a shape x size
# x colour x clarity grid is mostly empty cells, and an empty cell rendered as
# "slow" is a lie of omission.
_COLOR_TIERS = (("DEF", set("DEF")), ("GH", set("GH")), ("IJK", set("IJK")),
                ("LMN+", set("LMNOPQRSTUVWXYZ")))
_CLARITY_TIERS = (("FL-VVS", {"FL", "IF", "VVS1", "VVS2"}),
                  ("VS", {"VS1", "VS2"}),
                  ("SI", {"SI1", "SI2", "SI3"}),
                  ("I", {"I1", "I2", "I3", "P1", "P2", "P3"}))


def _tier(value: str, tiers) -> str:
    v = str(value or "").strip().upper()
    for name, members in tiers:
        if v in members:
            return name
    return "other"


def size_band_label(band: int) -> str:
    """Human-readable size band, read from SIZE_EDGES rather than hardcoded."""
    b = int(band)
    if b < 0 or b >= len(SIZE_EDGES):
        return "unknown"
    lo = SIZE_EDGES[b]
    if b + 1 < len(SIZE_EDGES):
        return f"{lo:.2f}-{SIZE_EDGES[b + 1] - 0.01:.2f}ct"
    return f"{lo:.2f}ct+"


def load_serving_model(frame: pd.DataFrame):
    """THE velocity model that answers a real request — the PROMOTED one.

    This function exists because the first build did not have it, and the
    consequence was CLAUDE.md Trap 5 in its fourth costume. `build_view()` fitted
    a fresh model on every cold start, so the nightly promotion gate governed
    nothing that reached a caller: had the gate REJECTED a bad candidate,
    serving would have refitted the rejected recipe and served it anyway. The
    gate would have looked like it was working — it writes a card, it logs a
    refusal — while protecting nothing.

    So: load what the gate promoted, and assert it is on the serving config, for
    the same reason `training/retrain.py` asserts it. Fitting fresh is a
    LAST-RESORT cold start (no model in the registry yet) and says so loudly,
    because an unaudited model answering client questions is exactly the state
    the registry exists to prevent.
    """
    from ..training.velocity_retrain import load_current
    # VelocityModel is imported HERE, not only in build_view(): the cold-start
    # branch below is the one path that constructs it, and it fires exactly when
    # there is no promoted model — i.e. on a fresh install, the first time anyone
    # runs this. `test_no_undefined_names` caught it before that happened.
    from .velocity import VelocityModel, serving_velocity_config

    try:
        model, card = load_current()
    except Exception:
        log.exception("could not read the velocity registry")
        model, card = None, None

    if model is not None:
        ref = serving_velocity_config()
        if getattr(model, "cfg", None) != ref:
            raise AssertionError(
                f"The promoted velocity model {card.get('version')} was trained on "
                f"a configuration that is not the one serving uses. Retrain before "
                f"serving from it — a gate that promotes a different model than "
                f"the one answering requests is not a gate.")
        log.info("Velocity: serving promoted model %s (C-index %s, calibration %s).",
                 card.get("version"), card.get("c_index"),
                 card.get("calibration_error"))
        return model, card

    log.warning("Velocity: NO promoted model in the registry — fitting an "
                "UNGATED model to answer this request. Run "
                "`python -m glowstar.training.velocity_retrain`. Until then "
                "nothing has checked this model against the segment-median "
                "baseline the client's screen already ships.")
    return VelocityModel(serving_velocity_config()).fit(frame), None


def _out_of_window_stock(records: pd.DataFrame, frame: pd.DataFrame,
                         rep) -> pd.DataFrame:
    """Stock the left-truncation guard removed, shaped for the ageing reports.

    Age and value only. Every velocity column is null and the basis says why —
    a number is never produced for these, and their absence is never silent.
    """
    stock = records[records["Status"] == "Stock"]
    seen = set(frame.loc[frame["Status"] == "Stock", "StoneId"].astype(str))
    out = stock[~stock["StoneId"].astype(str).isin(seen)]
    if not len(out) or rep.observation_asof is None:
        return out.head(0)
    age = (rep.observation_asof - out["MarketSheetDate_dt"]).dt.days
    basis = (f"listed before the sales window opened "
             f"({rep.window_start.date() if rep.window_start is not None else '?'}) "
             f"— age is reported, speed is not estimated (left-truncation, MOU 5.4)")
    return pd.DataFrame({
        "StoneId": out["StoneId"].astype(str).to_numpy(),
        "Segment": (out["Shape_full"].astype(str).str.title() + "|?|"
                    + out["Color"].astype(str).str.upper() + "|"
                    + out["Clarity"].astype(str).str.upper()).to_numpy(),
        "AgeDays": age.to_numpy(float),
        "AgeingBucket": [BF.ageing_bucket(a) for a in age],
        "ExpectedDaysToSell": np.nan, "ExpectedDaysLow": np.nan,
        "ExpectedDaysHigh": np.nan, "ExpectedTotalDays": np.nan,
        "OwnVelocityScore": np.nan, "HorizonLimited": False,
        "Class": None, "ClassFrontOffice": None,
        "SegmentSales": 0, "ThinSegment": True,
        "RedFlag": [BF.ageing_bucket(a) == BF.RED_FLAG_BUCKET for a in age],
        "MarketDepth": None, "MarketDepthScore": None,
        "MarketDepthBasis": "not looked up", "VelocityRatio": None,
        "OwnVsMarket": None, "OwnVsMarketBasis": basis,
        "ClassBasis": basis, "VelocityEstimated": False,
    })


@dataclass
class InventoryView:
    """Everything the dashboard and the reports need, computed once.

    Built here rather than in each consumer so the workbook and the endpoints
    can never disagree about what the book looks like — the same failure mode as
    the pricing engine's two configs, and cheaper to prevent than to find.
    """

    frame: pd.DataFrame
    stock: pd.DataFrame
    classified: pd.DataFrame
    segments: pd.DataFrame
    model: object
    report: object
    depth_table: object | None = None
    model_version: str | None = None
    velocity_card: dict | None = None
    generated_at: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def value_available(self) -> bool:
        return "StockValueUsd" in self.classified.columns


def build_view(*, records: pd.DataFrame | None = None, market=None,
               max_depth_segments: int = 0, model=None,
               depth: str = "banked") -> InventoryView:
    """Build the whole inventory picture once.

    `depth` selects where market depth comes from — and this choice was forced
    by measurement, not preference:

      * ``"banked"`` (default) reads `artifacts/market_segments.json`, which
        already holds deduped comparable counts for the whole book. Instant, and
        it carries its own age on every row.
      * ``"live"`` asks the feed per segment. Measured at **44 s per segment
        across 1,958 stock segments — roughly 24 hours a pass**, with 1 in 5
        failing. Usable for a handful of segments (set `max_depth_segments`),
        never for the book.
      * ``"none"`` skips depth entirely; every depth field then reads "not
        looked up" rather than zero, because a skipped feed must never render as
        "no competition", which scores WELL.

    Without this, MOU 5.2's own-vs-market ratio — an acceptance condition under
    10.3 — was null on every row of every payload and report.
    """
    from ..data.loaders import load_records
    from .market_depth import banked_depth_table, build_depth_table
    from .survival import build_survival_frame
    from .velocity import VelocityModel

    if records is None:
        records, _ = load_records()
    frame, rep = build_survival_frame(records)
    vcard = None
    if model is None:
        model, vcard = load_serving_model(frame)
    stock = frame[frame["Status"] == "Stock"].copy()

    depth_table = None
    if depth == "live" and market is not None and max_depth_segments > 0:
        depth_table = build_depth_table(stock, market=market,
                                        max_segments=max_depth_segments)
    elif depth == "banked":
        # The WHOLE frame, not just stock: `classify_segments` reports on every
        # segment the desk has traded, and building the table from stock alone
        # left 321 of the first 400 segments with no depth at all.
        depth_table = banked_depth_table(frame)
        if depth_table.n_requested == 0:
            depth_table = None

    classified = BF.classify_stones(stock, model, frame=frame, depth_table=depth_table)
    classified["VelocityEstimated"] = True
    segments = BF.classify_segments(frame, model, depth_table=depth_table)

    # THE OLDEST STOCK IS NOT IN THE VELOCITY FRAME, AND IT IS THE STOCK THE
    # DESK MOST WANTS TO SEE.
    #
    # The left-truncation guard excludes stones listed before the sales window
    # opened. That is correct for ESTIMATING velocity — those are survivors
    # whose contemporaries' sales are missing from the records — but it is
    # exactly wrong for REPORTING ageing, because the stones it removes are the
    # oldest ones on the book. Measured here: 1,189 stock stones, 10% of the
    # book by count and $2.3M of $13.1M by asking value, including 538 over a
    # year old and 83 over two years. The "365+" red-flag bucket came back EMPTY
    # while the client held 538 stones that belonged in it.
    #
    # So they are added back for anything that reports AGE or VALUE — which are
    # facts, not estimates — and carry `VelocityEstimated=False` so nothing
    # reports a speed for them. `_velocity_rows()` is what every velocity
    # payload filters on.
    excluded = _out_of_window_stock(records, frame, rep)
    if len(excluded):
        classified = pd.concat([classified, excluded], ignore_index=True)

    # Asking value per stone, joined from the records. It is the client's own
    # asking value (NetAmount), NOT cost — the feed carries no cost field — and
    # every payload that uses it says so.
    vals = (records[records["Status"] == "Stock"]
            .assign(_sid=lambda d: d["StoneId"].astype(str))
            .drop_duplicates("_sid").set_index("_sid")["NetAmount"])
    classified["StockValueUsd"] = pd.to_numeric(
        classified["StoneId"].map(vals), errors="coerce")

    notes = list(rep.notes)
    if len(excluded):
        notes.append(
            f"{len(excluded)} stock stones were listed before the sales window "
            f"opened. They are INCLUDED in the ageing and value figures (their "
            f"age is a fact) and EXCLUDED from every velocity figure (their "
            f"contemporaries' sales are not in the records, so a speed estimate "
            f"for them would be built on survivors only).")
    if depth_table is None:
        notes.append("market depth was not looked up in this run — the depth, "
                     "ratio and edge fields are null by design, not zero")
    else:
        notes.append(f"market depth resolved for {depth_table.coverage:.0%} of "
                     f"{depth_table.n_requested} stock segments"
                     + (" from the BANKED market snapshot (its age is on every "
                        "row's basis), because a live per-segment pull measures "
                        "at ~24 hours for this book" if depth == "banked" else
                        " from the live feed"))

    version = None
    try:
        from ..models import registry
        version = registry.current_version()
    except Exception:
        log.exception("could not read the pricing model version")

    return InventoryView(
        frame=frame, stock=stock, classified=classified, segments=segments,
        model=model, report=rep, depth_table=depth_table, model_version=version,
        velocity_card=vcard,
        generated_at=pd.Timestamp.now().isoformat(timespec="seconds"),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# limitations, carried in every payload
# ---------------------------------------------------------------------------
def _velocity_rows(view: InventoryView) -> pd.DataFrame:
    """Only stones with a velocity estimate. Every SPEED payload filters here.

    Ageing and value payloads deliberately do NOT: age and asking value are
    facts about every stone on the book, while a speed for a left-truncated
    stone would be built on survivors without their successes.
    """
    c = view.classified
    if "VelocityEstimated" not in c.columns:
        return c
    return c[c["VelocityEstimated"].astype(bool)]


def _clean(obj):
    """Recursively replace non-finite floats with None.

    NaN and Infinity are NOT valid JSON. Python will happily emit the bare
    tokens `NaN` / `Infinity`, and every strict parser — including the
    browser's own `JSON.parse` — throws on them. So a payload that looked fine
    from Python would have broken the first renderer to touch it, which is
    precisely the caller this MOU says we will not be writing ourselves.

    A missing number must therefore travel as `null`, which already means
    "not computed" everywhere in these payloads.
    """
    import math as _math

    if isinstance(obj, float):
        return None if not _math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if obj is pd.NaT:
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return None if isinstance(v, float) and not _math.isfinite(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def limitations(view: InventoryView) -> list[str]:
    """MOU 5.4 / 10.3: stated in the output, never buried in a document."""
    rep = view.report
    span = None
    if rep.window_start is not None and rep.observation_asof is not None:
        span = (rep.observation_asof - rep.window_start).days
    out = [
        (f"Sales history covers {span} days ({rep.window_start.date()} to "
         f"{rep.observation_asof.date()}). Under a year, so annual seasonality "
         f"is NOT learnable and none is claimed."
         if span is not None else "Sales history window unavailable."),
        ("Stones still in stock are RIGHT-CENSORED (they have not sold YET), and "
         "stones listed before the sales window opened are excluded as "
         "LEFT-TRUNCATED. Correcting one and not the other looks rigorous and is "
         f"worse than correcting neither; {rep.n_dropped_left_truncated} stones "
         "were excluded on that rule."),
        ("Days-to-sell is measured from MarketSheetDate, which is the client's "
         "own Ageing clock (verified exact on 100% of rows). AvailableDays is a "
         "different quantity and is not used."),
        ("Own velocity and market depth are reported separately and are never "
         "merged into one score (MOU 5.2)."),
        ("No causal effect of a price change on days-to-sell is claimed: none is "
         "identifiable from observational data here. See the repricing basis."),
        ("Estimates are read-and-recommend only. No price is applied and nothing "
         "is written to any client system (MOU 11.3, 11.9)."),
    ]
    out.extend(view.notes)
    return out


def _meta(view: InventoryView) -> dict:
    return {
        "generated_at": view.generated_at,
        "pricing_model_version": view.model_version,
        "velocity_model": {
            "method": "discrete-time hazard (HistGradientBoosting), "
                      "censoring handled natively",
            "trained_at": getattr(view.model, "trained_at", None),
            "train_listings": getattr(view.model, "n_train_stones", None),
            "train_sales": getattr(view.model, "n_train_events", None),
            "dropped_features": list(getattr(view.model, "dropped_features_", [])),
            # WHETHER THE MODEL ANSWERING THIS REQUEST PASSED THE GATE. A caller
            # must be able to tell an audited answer from a cold-start one; the
            # alternative is a payload that looks identical either way.
            "version": (view.velocity_card or {}).get("version"),
            "gate_passed": bool(view.velocity_card),
            "c_index": (view.velocity_card or {}).get("c_index"),
            "c_index_baseline": (view.velocity_card or {}).get("c_index_baseline"),
            "calibration_error": (view.velocity_card or {}).get("calibration_error"),
            "note": (None if view.velocity_card else
                     "UNGATED: no promoted velocity model was in the registry, so "
                     "this answer comes from a model fitted on the spot and "
                     "checked by nothing. Run glowstar.training.velocity_retrain."),
        },
        "observation_window": {
            "from": None if view.report.window_start is None else str(view.report.window_start.date()),
            "to": None if view.report.observation_asof is None else str(view.report.observation_asof.date()),
        },
        "class_vocabulary": {
            "canonical": list(BF.CLASSES),
            "frontoffice_equivalent": BF.FRONTOFFICE_LABELS,
            "note": "MOU 5.1 wording is canonical. The FrontOffice mapping is "
                    "kept until the client's screen is switched over (MOU 9.1).",
        },
        "limitations": limitations(view),
    }


# ---------------------------------------------------------------------------
# payloads
# ---------------------------------------------------------------------------
def stock_by_segment(view: InventoryView, limit: int = 200) -> dict:
    """Stock counts and asking value per segment, with each segment's basis."""
    c = _velocity_rows(view)
    g = c.groupby("Segment", observed=True)
    rows = pd.DataFrame({
        "segment": g.size().index,
        "stones": g.size().to_numpy(),
        "asking_value_usd": g["StockValueUsd"].sum().round(0).to_numpy(),
        "median_age_days": g["AgeDays"].median().round(0).to_numpy(),
        "median_expected_days_to_sell": g["ExpectedDaysToSell"].median().round(1).to_numpy(),
        "own_velocity_score": g["OwnVelocityScore"].median().round(0).to_numpy(),
        "own_sales_behind_it": g["SegmentSales"].max().to_numpy(),
    }).sort_values("asking_value_usd", ascending=False).head(limit)
    return _clean({
        "unit": {"asking_value_usd": "client's own asking value (NetAmount), "
                                     "NOT cost — the feed carries no cost field",
                 "days": "calendar days from MarketSheetDate"},
        "rows": rows.to_dict("records"),
        "meta": _meta(view),
    })


def velocity_heatmap(view: InventoryView, min_stones: int = 3) -> dict:
    """Fast <-> slow across shape x size x quality.

    Cells below `min_stones` are returned with `own_velocity_score: null` and a
    reason, never with a number. An empty cell rendered as "slow" would be a lie
    of omission, and the renderer has no way to know better.
    """
    c = _velocity_rows(view).copy()
    parts = c["Segment"].str.split("|", expand=True)
    c["shape"] = parts[0]
    c["size_band"] = pd.to_numeric(parts[1], errors="coerce").fillna(-1).astype(int)
    c["color_tier"] = [_tier(x, _COLOR_TIERS) for x in parts[2]]
    c["clarity_tier"] = [_tier(x, _CLARITY_TIERS) for x in parts[3]]
    c["quality"] = c["color_tier"] + " / " + c["clarity_tier"]

    cells = []
    for (shape, band, quality), g in c.groupby(["shape", "size_band", "quality"],
                                               observed=True):
        enough = len(g) >= min_stones
        cells.append({
            "shape": shape,
            "size_band": size_band_label(band),
            "quality": quality,
            "stones": int(len(g)),
            "own_velocity_score": (round(float(g["OwnVelocityScore"].median()))
                                   if enough else None),
            "class": (BF.serving_bifurcation_config().label(
                float(g["OwnVelocityScore"].median())) if enough else None),
            "median_expected_days_to_sell": (round(float(g["ExpectedDaysToSell"].median()), 1)
                                             if enough and g["ExpectedDaysToSell"].notna().any() else None),
            "asking_value_usd": round(float(g["StockValueUsd"].sum(skipna=True)), 0),
            "basis": (f"{len(g)} stones in stock, {int(g['SegmentSales'].max())} own sales behind the segment"
                      if enough else
                      f"only {len(g)} stone(s) — too few to score; shown for stock, not for speed"),
        })
    return _clean({
        "axes": {"x": "size_band", "y": "quality (colour tier / clarity tier)",
                 "panel": "shape"},
        "scale": {"own_velocity_score": "0-100, 100 = fastest goods THIS desk "
                                        "trades (percentile of its own book)"},
        "min_stones_to_score": min_stones,
        "cells": cells,
        "meta": _meta(view),
    })


def ageing_distribution(view: InventoryView) -> dict:
    """The four MOU buckets, by stone count and by asking value."""
    c = view.classified
    order = [b[2] for b in BF.AGEING_BUCKETS]
    rows = []
    total_val = float(c["StockValueUsd"].sum(skipna=True))
    for name in order:
        g = c[c["AgeingBucket"] == name]
        val = float(g["StockValueUsd"].sum(skipna=True))
        rows.append({
            "bucket": name,
            "stones": int(len(g)),
            "share_of_stones": round(len(g) / max(len(c), 1), 4),
            "asking_value_usd": round(val, 0),
            "share_of_value": round(val / total_val, 4) if total_val else None,
            "red_flag": name == BF.RED_FLAG_BUCKET,
        })
    return _clean({
        "buckets": rows,
        "red_flag_bucket": BF.RED_FLAG_BUCKET,
        "note": "Buckets are ABSOLUTE age. The five velocity classes are "
                "RELATIVE to this desk's own distribution. Read them together: "
                "'Slow and 200 days old' and 'Slow, listed last week' are "
                "different conversations.",
        "meta": _meta(view),
    })


def capital_at_risk(view: InventoryView,
                    slow_classes: tuple[str, ...] = ("Slow", "Semi-Slow")) -> dict:
    """Asking value sitting in slow movers, and in stale goods.

    Reported as the client's own asking value. It is NOT cost and NOT locked-up
    capital in the accounting sense — the feed carries no cost field — and the
    payload says so rather than letting a renderer label it "capital".
    """
    c = view.classified                      # WHOLE book: value and age are facts
    total = float(c["StockValueUsd"].sum(skipna=True))
    slow = c[c["Class"].isin(slow_classes)]  # only stones with a velocity class
    stale = c[c["AgeDays"] > 120]            # includes the left-truncated oldest
    both = c[c["Class"].isin(slow_classes) & (c["AgeDays"] > 120)]
    unscored = c[~c["VelocityEstimated"].astype(bool)] if "VelocityEstimated" in c else c.head(0)

    def block(g, label):
        v = float(g["StockValueUsd"].sum(skipna=True))
        return {"label": label, "stones": int(len(g)),
                "asking_value_usd": round(v, 0),
                "share_of_book": round(v / total, 4) if total else None}

    return _clean({
        "total_stock_asking_value_usd": round(total, 0),
        "unit_note": "client's own asking value (NetAmount). NOT cost: the feed "
                     "carries no cost field, so true capital-at-risk and GMROI "
                     "cannot be computed from it.",
        "slow_classes": list(slow_classes),
        "breakdown": [
            block(slow, f"in {' or '.join(slow_classes)}"),
            block(stale, "older than 120 days"),
            block(both, "slow AND older than 120 days — the real accumulation"),
            block(unscored, "listed before the sales window — age reported, "
                            "speed not estimated (NOT counted in 'slow' above)"),
        ],
        "by_bucket": ageing_distribution(view)["buckets"],
        "meta": _meta(view),
    })


def segment_detail(view: InventoryView, limit: int = 200) -> dict:
    """Per segment: own velocity vs market depth, days-to-sell, and the gap.

    The two numbers stay two numbers. `velocity_ratio` is derived and labelled
    as derived; there is no blended score anywhere in this payload.
    """
    s = view.segments.head(limit).copy()
    rows = []
    for r in s.to_dict("records"):
        rows.append({
            "segment": r["Segment"],
            "class": r["Class"],
            "class_frontoffice": r["ClassFrontOffice"],
            "stones_total": int(r["Stones"]),
            "stones_in_stock": int(r["InStock"]),
            "own_sales": int(r["SegmentSales"]),
            "expected_days_to_sell": (None if pd.isna(r["ExpectedDaysToSell"])
                                      else float(r["ExpectedDaysToSell"])),
            "own_velocity_score": (None if pd.isna(r["OwnVelocityScore"])
                                   else int(r["OwnVelocityScore"])),
            "market_depth": r.get("MarketDepth"),
            "market_depth_score": r.get("MarketDepthScore"),
            "velocity_ratio": r.get("VelocityRatio"),
            "own_vs_market": r.get("OwnVsMarket"),
            "thin_segment": bool(r["ThinSegment"]),
            "basis": r["Basis"],
        })
    return _clean({
        "note": "own_velocity_score and market_depth_score are SEPARATE "
                "measures and must stay separate (MOU 5.2). velocity_ratio is "
                "own/market and is derived from them, not a third measurement.",
        "rows": rows,
        "meta": _meta(view),
    })


def dashboard(view: InventoryView) -> dict:
    """Everything, one payload, each section self-describing."""
    return {
        "stock_by_segment": stock_by_segment(view),
        "velocity_heatmap": velocity_heatmap(view),
        "ageing_distribution": ageing_distribution(view),
        "capital_at_risk": capital_at_risk(view),
        "segment_detail": segment_detail(view),
    }


def main() -> None:
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    view = build_view()
    car = capital_at_risk(view)
    print(json.dumps({k: v for k, v in car.items() if k != "meta"},
                     indent=2, default=str))
    hm = velocity_heatmap(view)
    scored = [c for c in hm["cells"] if c["own_velocity_score"] is not None]
    print(f"\nheatmap: {len(hm['cells'])} cells, {len(scored)} with enough stock to score")
    print("\nlimitations carried in every payload:")
    for line in limitations(view):
        print(" -", line)


if __name__ == "__main__":
    main()
