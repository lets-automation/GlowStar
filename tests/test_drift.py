"""Drift alarm: the after-the-fact check that published prices track reality."""

from __future__ import annotations

import pandas as pd
import pytest

from glowstar.monitoring import drift as D


def _dec(**kw):
    base = {"ts": pd.Timestamp("2026-08-11"), "decision": "override",
            "suggested_discount": -50.0, "human_discount": -55.0,
            "reason_code": "market_moved"}
    base.update(kw)
    return base


def test_empty_store_is_not_an_error():
    """No data must mean 'nothing to say', never a crash or a false alarm."""
    w, r = D.override_drift(pd.DataFrame())
    assert w.empty and r.empty
    g, n = D.realized_drift(pd.DataFrame())
    assert g.empty and n == 0


def test_override_without_a_desk_price_is_excluded():
    """A reason with no number records that we were wrong, not what right is.

    Counting it would silently treat 'no opinion' as 'zero variance' and drag
    the average toward zero — hiding the very drift this exists to surface.
    """
    d = pd.DataFrame([_dec(), _dec(human_discount=None), _dec(human_discount=None)])
    weekly, _ = D.override_drift(d)
    assert int(weekly["n"].sum()) == 1


def test_override_sign_is_preserved_not_absolute():
    """Direction is the point: consistently positive = systematically shallow."""
    d = pd.DataFrame([_dec(suggested_discount=-50.0, human_discount=-56.0),
                      _dec(suggested_discount=-50.0, human_discount=-56.0)])
    weekly, _ = D.override_drift(d)
    assert weekly["mean_signed"].iloc[0] == pytest.approx(6.0)
    assert weekly["mean_abs"].iloc[0] == pytest.approx(6.0)
    # opposite direction must cancel in the signed mean, not in the absolute
    d2 = pd.DataFrame([_dec(human_discount=-56.0), _dec(human_discount=-44.0)])
    w2, _ = D.override_drift(d2)
    assert w2["mean_signed"].iloc[0] == pytest.approx(0.0)
    assert w2["mean_abs"].iloc[0] == pytest.approx(6.0)


def test_accepts_and_rejects_are_not_counted_as_overrides():
    d = pd.DataFrame([_dec(decision="accept"), _dec(decision="reject"), _dec()])
    weekly, _ = D.override_drift(d)
    assert int(weekly["n"].sum()) == 1


def test_alert_fires_only_with_enough_rows():
    """One furious stone is not a trend; MIN_ROWS guards against alarm fatigue."""
    big = pd.DataFrame([_dec(human_discount=-70.0)] * D.MIN_ROWS)      # +20 pts each
    weekly, _ = D.override_drift(big)
    assert abs(weekly["mean_signed"].iloc[0]) > D.OVERRIDE_VARIANCE_ALERT
    assert int(weekly["n"].iloc[0]) >= D.MIN_ROWS

    few = pd.DataFrame([_dec(human_discount=-70.0)] * 2)
    w2, _ = D.override_drift(few)
    assert int(w2["n"].iloc[0]) < D.MIN_ROWS       # would NOT alert


def test_report_never_raises_and_reports_alert_state():
    r = D.build_report()
    assert isinstance(r.alerts, list)
    assert isinstance(D.format_report(r), str)
    assert r.ok() == (not r.alerts)
