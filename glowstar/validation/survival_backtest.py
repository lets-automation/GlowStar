"""Out-of-time validation for the velocity model — Workstream B, Phase B gate.

THE PROTOCOL, AND WHY IT IS THE ONE THAT SHIPS
----------------------------------------------
The question the model answers is "how long will this stone take to sell?",
asked ON THE DAY IT IS LISTED. So the split is on the LISTING date, not the sale
date, and the training set is ADMINISTRATIVELY CENSORED at the split:

    train : stones listed before `split`, with everything that happened after
            `split` erased — a stone that sold on `split + 10` is recorded as
            still unsold, because on the split date that is what we knew.
    test  : stones listed on or after `split`, scored against what they actually
            did.

Without the administrative censoring the "training" set would contain outcomes
from the future of its own split, the model would be reading the answer, and the
C-index would look excellent while measuring nothing. That is CLAUDE.md Trap 2
in survival clothing, and it is the single easiest thing to get wrong here.

THE BAR
-------
The model must beat the SEGMENT-MEDIAN baseline out-of-time. That baseline is
essentially what `service/tradeability.py` ships today, so this is not a straw
man: passing means the model is worth binding the live FrontOffice field to.

KNOWN LIMIT, STATED RATHER THAN BURIED (MOU 5.4)
-------------------------------------------------
Test stones are only followed to the current snapshot, so late in the window
follow-up is short and the test set is heavily censored. The C-index handles
that correctly — it only scores pairs whose ordering is actually known — but it
means the metric is dominated by the fast half of the book. Calibration at 30
and 60 days is therefore reported alongside it, and the 90-day column thins out.

Run:  python -m glowstar.validation.survival_backtest
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import SETTINGS
from ..inventory import survival as S
from ..inventory.velocity import (SegmentMedianBaseline, VelocityModel,
                                  serving_velocity_config)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harrell's C for right-censored data
# ---------------------------------------------------------------------------
class _BIT:
    """Fenwick tree over risk ranks — the whole reason this is not O(n^2).

    16k test stones is 256M pairs; counted directly in Python that is minutes
    per evaluation and the nightly gate would simply not run.
    """

    __slots__ = ("n", "t")

    def __init__(self, n: int):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i: int) -> None:
        i += 1
        while i <= self.n:
            self.t[i] += 1
            i += i & (-i)

    def prefix(self, i: int) -> int:
        """Count of entries with rank < i."""
        total = 0
        while i > 0:
            total += self.t[i]
            i -= i & (-i)
        return total

    def total(self) -> int:
        return self.prefix(self.n)


def concordance_index(durations, events, risk) -> tuple[float, int]:
    """Harrell's C for right-censored data. Returns (c_index, comparable_pairs).

    `risk` is HIGHER for a stone expected to sell SOONER, so it is the negative
    of expected days. Concordant means the stone we ranked riskier really did
    sell first.

    Comparability, spelled out because the tie rules are where implementations
    quietly disagree:
      * (i, j) is comparable when i SOLD and `T_i < T_j` — whatever happened to
        j, it had not sold by `T_i`, so the ordering is known;
      * also comparable when i sold, j is censored, and `T_i == T_j` — j left
        the book unsold on the day i sold;
      * NOT comparable when both sold at the same time — the ordering is
        genuinely unknown and guessing it is how a C-index gets inflated.
    Tied risks score 0.5, the standard convention.

    Returns `(nan, 0)` when nothing is comparable, rather than a made-up 0.5.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events).astype(bool)
    r = np.asarray(risk, dtype=float)
    if len(t) == 0:
        return float("nan"), 0

    # Non-finite risks cannot be ranked; drop them and say so via the pair count.
    ok = np.isfinite(r) & np.isfinite(t)
    t, e, r = t[ok], e[ok], r[ok]
    if len(t) == 0:
        return float("nan"), 0

    uniq = np.unique(r)
    rank = np.searchsorted(uniq, r)             # 0-based dense ranks
    n_ranks = len(uniq)

    order = np.argsort(-t, kind="mergesort")    # descending time
    t_s, e_s, rank_s = t[order], e[order], rank[order]

    bit = _BIT(n_ranks)
    concordant = tied = 0.0
    comparable = 0
    i = 0
    n = len(t_s)
    while i < n:
        j = i
        while j < n and t_s[j] == t_s[i]:
            j += 1
        grp = slice(i, j)
        ev_ranks = rank_s[grp][e_s[grp]]
        cen_ranks = np.sort(rank_s[grp][~e_s[grp]])

        for rk in ev_ranks:
            # pairs against everything with a strictly LATER time
            later = bit.total()
            comparable += later
            concordant += bit.prefix(rk)                 # ranks strictly below
            tied += bit.prefix(rk + 1) - bit.prefix(rk)  # ranks exactly equal
        # pairs against stones CENSORED at this very time
        if len(ev_ranks) and len(cen_ranks):
            lo = np.searchsorted(cen_ranks, ev_ranks, side="left")
            hi = np.searchsorted(cen_ranks, ev_ranks, side="right")
            comparable += int(len(cen_ranks) * len(ev_ranks))
            concordant += float(lo.sum())
            tied += float((hi - lo).sum())

        for rk in rank_s[grp]:
            bit.add(int(rk))
        i = j

    if comparable == 0:
        return float("nan"), 0
    return float((concordant + 0.5 * tied) / comparable), int(comparable)


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------
def out_of_time_split(frame: pd.DataFrame, split_date: str
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Listing-date split with the training arm censored at `split_date`.

    See the module docstring: erasing post-split outcomes from the training set
    is what makes this a forecast rather than a look-up.
    """
    split = pd.Timestamp(split_date)
    train = frame[frame["entered"] < split].copy()
    test = frame[frame["entered"] >= split].copy()

    available = (split - train["entered"]).dt.days.to_numpy(float)
    sold_by_split = (train["event"].to_numpy(int) == 1) & (train["duration"].to_numpy(float) <= available)
    train["event"] = sold_by_split.astype(int)
    train["duration"] = np.minimum(train["duration"].to_numpy(float), available)
    return train, test


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def calibration_table(frame: pd.DataFrame, p_sold: np.ndarray, horizon: float,
                      n_bins: int = 5) -> pd.DataFrame:
    """Predicted vs OBSERVED P(sold by `horizon`), by predicted-probability bin.

    The observed side is `1 - KM(horizon)` inside each bin, not a raw sold-count:
    a stone listed 20 days ago has not had the chance to sell by day 60, and
    counting it as a failure would make every model look badly over-confident.
    Bins whose curve cannot reach the horizon are reported as `n_at_risk=0`
    rather than given a number.
    """
    p = np.asarray(p_sold, dtype=float)
    dur = frame["duration"].to_numpy(float)
    ev = frame["event"].to_numpy(int).astype(bool)

    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for b in range(n_bins):
        m = (p > edges[b]) & (p <= edges[b + 1])
        if not m.any():
            continue
        curve = S.km_curve(dur[m], ev[m])
        followed = int((dur[m] >= horizon).sum())
        observed = (1.0 - curve.survival_at(horizon)) if followed > 0 else np.nan
        rows.append({
            "bin": b + 1,
            "n": int(m.sum()),
            "n_followed_to_horizon": followed,
            "predicted": round(float(p[m].mean()), 3),
            "observed": None if np.isnan(observed) else round(float(observed), 3),
            "gap": None if np.isnan(observed) else round(float(p[m].mean() - observed), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def evaluate(frame: pd.DataFrame | None = None, split_date: str | None = None,
             *, cfg=None) -> dict:
    """Fit on the past, score on the future. The Phase-B gate in one call."""
    if frame is None:
        frame, _ = S.build_survival_frame()
    split_date = split_date or SETTINGS.backtest_split_date
    cfg = cfg or serving_velocity_config()

    train, test = out_of_time_split(frame, split_date)
    if not len(test) or not int(test["event"].sum()):
        raise ValueError(f"no test events after {split_date} — move the split earlier")

    model = VelocityModel(cfg).fit(train)
    pred = model.predict_days(test)
    # risk: sooner = riskier. A stone whose median is never reached is the
    # slowest thing on the book, so it gets the lowest finite risk rather than
    # being dropped from the ranking.
    days = pred["expected_days"].to_numpy(float)
    risk = -np.where(np.isfinite(days), days, model.cfg.period_edges[-1] * 10.0)

    base = SegmentMedianBaseline(min_sales=cfg.min_segment_sales).fit(train)
    base_days, _ = base.predict(test)

    c_model, pairs = concordance_index(test["duration"], test["event"].astype(bool), risk)
    c_base, _ = concordance_index(test["duration"], test["event"].astype(bool), -base_days)

    surv = model.predict_survival(test)
    result = {
        "split_date": split_date,
        "n_train": int(len(train)), "n_train_events": int(train["event"].sum()),
        "n_test": int(len(test)), "n_test_events": int(test["event"].sum()),
        "comparable_pairs": pairs,
        "c_index_model": None if np.isnan(c_model) else round(c_model, 4),
        "c_index_segment_median": None if np.isnan(c_base) else round(c_base, 4),
        "beats_baseline": bool(c_model > c_base),
        "median_test_followup_days": int((test["duration"]).median()),
        "calibration": {},
    }
    for h in (30.0, 60.0, 90.0):
        p_sold = 1.0 - model._survival_at(surv, h)
        result["calibration"][int(h)] = calibration_table(test, p_sold, h).to_dict("records")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = evaluate()
    print("\n" + "=" * 70)
    print("VELOCITY MODEL — OUT-OF-TIME (listing-date split, train censored at split)")
    print("=" * 70)
    print(f"split              : {r['split_date']}")
    print(f"train              : {r['n_train']:,} listings, {r['n_train_events']:,} sold by the split")
    print(f"test               : {r['n_test']:,} listings, {r['n_test_events']:,} sold since")
    print(f"comparable pairs   : {r['comparable_pairs']:,}")
    print(f"C-index  model     : {r['c_index_model']}")
    print(f"C-index  baseline  : {r['c_index_segment_median']}   (segment median — what ships today)")
    print(f"beats the baseline : {r['beats_baseline']}")
    for h, rows in r["calibration"].items():
        print(f"\ncalibration at {h} days (predicted vs observed P(sold), KM-corrected):")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nNote: test stones are followed only to the current snapshot, so the")
    print("90-day column is thin. A wide or absent number there is the honest one.")


if __name__ == "__main__":
    main()
