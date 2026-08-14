"""Drift alarm — does what we PUBLISHED match what actually happened?

WHY THIS EXISTS
---------------
Nothing in this system compared published prices against reality after the
fact. Every quote is recorded in `quotes`; every stone that sells appears in
`records.json` weeks later; nothing joined them. That gap is how a frozen grid
feed went unnoticed for fifteen days — the model kept answering, `/health` kept
saying 200, and the only symptom was a number drifting upward that nobody was
watching.

Two independent signals, because they fail differently:

  * REALIZED drift — quotes joined to the sale that actually happened. The
    ground truth, but it arrives weeks late.
  * OVERRIDE drift — what the desk corrected us by, available immediately.
    Biased (they only correct what looks wrong to them) and their price is an
    ASKING quote, not a sale — so it is a leading indicator, never a target.
    Read it as "where is the desk unhappy", not "what is the right price".

Neither is reported as a single number. Both are broken down, because an
average hides the thing you need: a segment that has gone wrong while the book
as a whole looks fine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import select

log = logging.getLogger(__name__)

# A weekly realized MAE above this is a real problem, not noise. Set from the
# measured out-of-time figure (~2.0-2.4) plus room for a hard week.
REALIZED_MAE_ALERT = 3.5

# The desk correcting us by more than this ON AVERAGE means something
# systematic, not one awkward stone.
OVERRIDE_VARIANCE_ALERT = 5.0

# Below this many rows a "trend" is noise. Report it, never alert on it.
MIN_ROWS = 10


@dataclass
class DriftReport:
    realized: pd.DataFrame = field(default_factory=pd.DataFrame)
    overrides: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_reason: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: list[str] = field(default_factory=list)
    n_quotes: int = 0
    n_matched: int = 0
    n_overrides: int = 0

    def ok(self) -> bool:
        return not self.alerts


def _quotes_frame() -> pd.DataFrame:
    from ..store.db import get_engine, quotes
    with get_engine().connect() as c:
        rows = c.execute(select(quotes)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


def _decisions_frame() -> pd.DataFrame:
    from ..store.db import get_engine, decisions
    with get_engine().connect() as c:
        rows = c.execute(select(decisions)).mappings().all()
    return pd.DataFrame([dict(r) for r in rows])


def realized_drift(q: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Join published quotes to the sale that actually happened.

    Matching is by certificate number first (globally unique) and stone id
    second. A quote with no matching sale is NOT an error — the stone may still
    be in stock, or belong to someone else's book. It is simply not yet
    evidence, so it is excluded rather than counted as a miss.
    """
    from ..data.loaders import load_records, sold_stones

    if q.empty:
        return pd.DataFrame(), 0
    sold = sold_stones(load_records()[0], drop_outliers=True)

    sold_c = sold.assign(_k=sold.get("CertificateNo", pd.Series(dtype=str)).astype(str))
    sold_s = sold.assign(_k=sold["StoneId"].astype(str))
    truth = pd.concat([sold_c, sold_s], ignore_index=True)
    truth = truth[truth["_k"].notna() & (truth["_k"] != "") & (truth["_k"] != "nan")]
    truth = truth.drop_duplicates("_k", keep="last")[["_k", "FDiscount", "OrderDate_dt"]]

    q = q.copy()
    q["_k"] = q["certificate_no"].astype(str)
    m = q.merge(truth, on="_k", how="inner")
    if m.empty:
        m = q.assign(_k=q["stone_id"].astype(str)).merge(truth, on="_k", how="inner")
    if m.empty:
        return pd.DataFrame(), 0

    # Only sales that happened AFTER we quoted. A sale that predates the quote
    # tells us nothing about that quote and would flatter the numbers.
    m["ts"] = pd.to_datetime(m["ts"], utc=True, errors="coerce").dt.tz_localize(None)
    m = m[m["OrderDate_dt"] >= m["ts"].dt.normalize()]
    if m.empty:
        return pd.DataFrame(), 0

    m["abs_err"] = (m["discount"] - m["FDiscount"]).abs()
    m["week"] = m["ts"].dt.to_period("W").astype(str)
    g = m.groupby("week").agg(
        stones=("abs_err", "size"),
        mae=("abs_err", "mean"),
        within2=("abs_err", lambda s: float((s <= 2).mean())),
        within5=("abs_err", lambda s: float((s <= 5).mean())),
    ).round(3).reset_index()
    return g, len(m)


def override_drift(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What the desk corrected us by — the leading indicator.

    Only rows carrying the desk's OWN price are usable. A reason with no number
    records that we were wrong, never what right looks like.
    """
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = d[d["decision"].astype(str).str.lower() == "override"].copy()
    d = d[d["human_discount"].notna()]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce").dt.tz_localize(None)
    d["week"] = d["ts"].dt.to_period("W").astype(str)
    # SIGNED, not absolute: the direction is the whole point. Consistently
    # positive means we are systematically too shallow, and that is fixable in
    # a way that a symmetric error is not.
    d["signed"] = d["suggested_discount"] - d["human_discount"]

    weekly = d.groupby("week").agg(
        n=("signed", "size"),
        mean_signed=("signed", "mean"),
        mean_abs=("signed", lambda s: float(s.abs().mean())),
    ).round(2).reset_index()

    by_reason = (d.groupby(d["reason_code"].fillna("(none)"))
                 .agg(n=("signed", "size"),
                      mean_signed=("signed", "mean"),
                      mean_abs=("signed", lambda s: float(s.abs().mean())))
                 .round(2).sort_values("n", ascending=False).reset_index())
    by_reason.columns = ["reason", "n", "mean_signed", "mean_abs"]
    return weekly, by_reason


def build_report() -> DriftReport:
    r = DriftReport()
    try:
        q = _quotes_frame()
        d = _decisions_frame()
    except Exception:
        log.exception("drift: store unavailable")
        r.alerts.append("store unavailable — drift could not be computed")
        return r

    r.n_quotes = len(q)
    r.realized, r.n_matched = realized_drift(q)
    r.overrides, r.by_reason = override_drift(d)
    r.n_overrides = 0 if r.overrides.empty else int(r.overrides["n"].sum())

    if not r.realized.empty:
        last = r.realized.iloc[-1]
        if last["stones"] >= MIN_ROWS and last["mae"] > REALIZED_MAE_ALERT:
            r.alerts.append(
                f"realized MAE {last['mae']} in week {last['week']} "
                f"(> {REALIZED_MAE_ALERT}) on {int(last['stones'])} stones")

    if not r.overrides.empty:
        last = r.overrides.iloc[-1]
        if last["n"] >= MIN_ROWS and abs(last["mean_signed"]) > OVERRIDE_VARIANCE_ALERT:
            direction = "too shallow" if last["mean_signed"] > 0 else "too deep"
            r.alerts.append(
                f"desk overrode {int(last['n'])} stones in week {last['week']} "
                f"averaging {last['mean_signed']:+.2f} pts — we are systematically "
                f"{direction}")
    return r


def format_report(r: DriftReport) -> str:
    out = ["=" * 62, "DRIFT REPORT", "=" * 62,
           f"quotes published : {r.n_quotes:,}",
           f"matched to a sale: {r.n_matched:,}"
           + ("  (none yet — sales arrive weeks later)" if not r.n_matched else ""),
           f"desk overrides   : {r.n_overrides:,}", ""]
    if not r.realized.empty:
        out += ["REALIZED (published price vs the sale that happened)",
                r.realized.to_string(index=False), ""]
    if not r.overrides.empty:
        out += ["OVERRIDES (desk's own price; ASKING quote, not a sale)",
                r.overrides.to_string(index=False), ""]
    if not r.by_reason.empty:
        out += ["BY REASON", r.by_reason.to_string(index=False), ""]
    out += (["ALERTS"] + [f"  !! {a}" for a in r.alerts]) if r.alerts else ["no alerts"]
    return "\n".join(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = build_report()
    print(format_report(r))
    for a in r.alerts:
        log.warning("DRIFT ALERT: %s", a)


if __name__ == "__main__":
    main()
