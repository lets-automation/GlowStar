"""End-to-end tests for the two doors the client's CRM talks to.

These drive the real app through a test client — request in, JSON out — because
the failure that matters is "the CRM sent us something and we broke", not "the
function returns the right type".
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient        # noqa: E402

from glowstar.service import app as app_mod      # noqa: E402
from glowstar.feedback.store import VARIANCE_REASON_THRESHOLD_PTS  # noqa: E402


STONE = {
    "StoneId": "TEST-1", "Shape_full": "Round", "Weight": 0.52, "Color": "G",
    "Clarity": "VS1", "CPS": "3EX", "Fluorescence": "Non", "Lab": "GIA",
    "Location": "IND", "Rap": 1900.0,
}


class _FakeEngineCfg:
    coverage = 0.8


class _FakeSuggestion:
    """Minimal stand-in so the API tests do not depend on a trained model."""
    stone_id = "TEST-1"
    suggested_discount = -50.0
    suggested_ppc = 950.0
    suggested_net = 494.0
    ci_discount_low, ci_discount_high = -54.0, -46.0
    ci_net_low, ci_net_high = 455.0, 533.0
    comparable_count = 120
    market_median_discount = -48.0
    method = "model+anchor"
    flags: list = []
    market_direction = "flat"
    trend_shift_pts = 0.0
    bgm_state = "clean"
    bgm_deduction_pts = 0.0
    assumes_no_bgm = False
    feedback_correction_pts = 0.0


class _FakeService:
    cfg = _FakeEngineCfg()

    class _Eng:
        def set_corrections(self, *_a, **_k):
            pass
    engine = _Eng()

    def price(self, stone):
        return {"suggestion": {"stone_id": stone.StoneId,
                               "suggested_discount": -50.0,
                               "suggested_ppc": 950.0},
                "market": {}, "explanation": "test"}

    # Mirrors the REAL PricingService.price_many contract: one entry per stone,
    # in order, and a stone that fails comes back AS an Exception rather than
    # raising — that is what lets one bad stone cost only its own row.
    # `test_price_many_returns_failures_it_does_not_raise_them` pins the real
    # service to this same shape so the double cannot quietly drift from it.
    def price_many(self, stones, *, explain=True):
        out = []
        for s in stones:
            try:
                out.append(self.price(s))
            except Exception as e:      # noqa: BLE001 - contract is to return it
                out.append(e)
        return out


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """App wired to a fake service and a throwaway feedback log."""
    monkeypatch.setattr(app_mod, "_service", _FakeService())
    import glowstar.feedback.store as store
    monkeypatch.setattr(store, "FEEDBACK_LOG", tmp_path / "decisions.jsonl")
    monkeypatch.delenv("GS_API_KEY", raising=False)
    return TestClient(app_mod.app)


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------
def test_health_reports_what_is_actually_serving(client):
    """A 200 that hides a stale model is not health."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "model" in body and "records_age_hours" in body


# --------------------------------------------------------------------------
# door 1 — price
# --------------------------------------------------------------------------
def test_price_one_stone(client):
    r = client.post("/price", json=STONE)
    assert r.status_code == 200
    assert r.json()["suggestion"]["suggested_discount"] == -50.0


def test_price_rejects_an_impossible_stone(client):
    bad = dict(STONE, Weight=-1)
    assert client.post("/price", json=bad).status_code == 422


def test_batch_isolates_a_bad_stone_instead_of_failing_the_book(client, monkeypatch):
    """One bad stone must never cost the desk the other 999 prices."""
    class _Flaky(_FakeService):
        def price(self, stone):
            if stone.StoneId == "BOOM":
                raise RuntimeError("no market data")
            return super().price(stone)
    monkeypatch.setattr(app_mod, "_service", _Flaky())

    r = client.post("/price/batch", json=[STONE, dict(STONE, StoneId="BOOM"), STONE])
    assert r.status_code == 200
    body = r.json()
    assert body["n_priced"] == 2 and body["n_failed"] == 1
    assert body["failed"][0]["stone_id"] == "BOOM"
    assert "no market data" in body["failed"][0]["error"]


# --------------------------------------------------------------------------
# door 2 — decision + the variance threshold
# --------------------------------------------------------------------------
def _decision(**kw):
    base = {"stone_id": "TEST-1", "decision": "override",
            "suggested_discount": -50.0, "suggested_net": 494.0,
            "shape_full": "Round", "weight": 0.52, "color": "G", "clarity": "VS1",
            "rap": 1900.0}
    base.update(kw)
    return base


def test_small_override_needs_no_reason(client):
    """Client rule: inside the threshold, the desk just types a price."""
    r = client.post("/decision", json=_decision(human_discount=-51.0))
    assert r.status_code == 200
    body = r.json()
    assert body["recorded"] is True
    assert body["variance_pts"] == 1.0
    assert body["needs_attention"] is False
    assert body["reason_required"] is False
    assert body["threshold_pts"] == VARIANCE_REASON_THRESHOLD_PTS


def test_large_override_without_a_reason_is_refused(client):
    r = client.post("/decision", json=_decision(human_discount=-58.0))
    assert r.status_code == 422
    assert "reason_code" in r.json()["detail"]


def test_large_override_with_a_reason_is_recorded_and_flagged(client):
    r = client.post("/decision",
                    json=_decision(human_discount=-58.0, reason_code="market_moved"))
    assert r.status_code == 200
    body = r.json()
    assert body["variance_pts"] == 8.0
    assert body["needs_attention"] is True


def test_crm_may_send_dollars_instead_of_a_discount(client):
    """The CRM works in $/ct; it must never have to do pricing arithmetic."""
    # 1900 Rap, $912/ct -> -52.0% discount, i.e. 2 pts from our -50
    r = client.post("/decision", json=_decision(human_ppc=912.0, human_discount=None))
    assert r.status_code == 200
    assert r.json()["variance_pts"] == pytest.approx(2.0, abs=0.01)


def test_ppc_without_rap_is_a_clear_error_not_a_wrong_number(client):
    r = client.post("/decision",
                    json=_decision(human_ppc=912.0, human_discount=None, rap=0.0))
    assert r.status_code == 422
    assert "rap" in r.json()["detail"].lower()


def test_unknown_decision_is_refused(client):
    r = client.post("/decision", json=_decision(decision="maybe", human_discount=-51.0))
    assert r.status_code == 422


def test_override_without_a_price_is_refused(client):
    """No price = no label = the record teaches nothing."""
    r = client.post("/decision", json=_decision(human_discount=None))
    assert r.status_code == 422


def test_accept_is_recorded_and_persisted(client, tmp_path):
    r = client.post("/decision", json={"stone_id": "TEST-9", "decision": "accept",
                                       "suggested_discount": -50.0})
    assert r.status_code == 200
    log = tmp_path / "decisions.jsonl"
    assert log.exists()
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert rows[-1]["stone_id"] == "TEST-9" and rows[-1]["decision"] == "accept"


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------
def test_api_key_is_enforced_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "_service", _FakeService())
    import glowstar.feedback.store as store
    monkeypatch.setattr(store, "FEEDBACK_LOG", tmp_path / "d.jsonl")
    monkeypatch.setenv("GS_API_KEY", "secret123")
    c = TestClient(app_mod.app)

    assert c.post("/price", json=STONE).status_code == 401
    assert c.post("/price", json=STONE, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/price", json=STONE,
                  headers={"X-API-Key": "secret123"}).status_code == 200
    # health stays open so a load balancer can probe it
    assert c.get("/health").status_code == 200


# --- a stone we cannot price must not take down the whole run ---------------
# PRODUCTION, 2026-08-21: the client selected 595 stones. Two were "Fancy Intense
# Yellow", which has no cell on the white D-N Rapaport list, so the lookup raised
# and /price returned a bare 500. Their CRM rendered it as `[object Object]` and
# abandoned the entire run — 593 priceable stones lost to two yellows.

def test_fancy_colour_returns_a_structured_422_not_a_500(monkeypatch):
    from fastapi.testclient import TestClient
    from glowstar.service import app as app_mod
    monkeypatch.delenv("GS_API_KEY", raising=False)
    c = TestClient(app_mod.app)
    r = c.post("/price", json={"Shape_full": "Cushion", "Weight": 1.03,
                               "Color": "Fancy Intense Yellow", "Clarity": "VVS2"})
    assert r.status_code == 422, "an unpriceable stone must not be a server error"
    d = r.json()["detail"]
    assert d["error"] == "not_priceable"
    # the caller must be told WHICH stone and WHY, or they cannot skip it
    assert d["color"] == "Fancy Intense Yellow"
    assert "white" in d["message"].lower()
    assert "frontoffice" in d["hint"]


def test_a_batch_survives_a_stone_it_cannot_price(monkeypatch):
    """The other 593 must still come back."""
    from fastapi.testclient import TestClient
    from glowstar.service import app as app_mod
    monkeypatch.delenv("GS_API_KEY", raising=False)
    c = TestClient(app_mod.app)
    r = c.post("/frontoffice/price", json=[
        {"stoneId": "FANCY", "shape": "Cushion", "weight": 1.03,
         "color": "Fancy Intense Yellow", "clarity": "VVS2"},
        {"stoneId": "WHITE", "shape": "RBC", "weight": 0.33,
         "color": "F", "clarity": "VVS1", "fluorescence": "NON"},
    ])
    assert r.status_code == 200
    rows = {x["StoneId"]: x for x in r.json()}
    assert rows["FANCY"]["AIDiscount"] is None and rows["FANCY"]["Error"]
    assert rows["WHITE"]["AIDiscount"] is not None, "a good stone must still price"


# --- /health must describe what is SERVING, not what is on disk -------------
# The model is loaded once per uvicorn worker and there is no reload path, so a
# nightly promotion does not reach a running worker — only a restart does.
# /health called registry.load_current() and reported the promoted version as
# though it were live, so a worker could serve a months-old model behind a green
# health check. CLAUDE.md Trap 5: never describe a pipeline the client is not
# being served by. (Trap 9's rule too: /health is not evidence pricing is right.)

class _Loaded:
    def __init__(self, version): self.model_version = version


def test_health_flags_a_worker_serving_a_stale_model(client, monkeypatch):
    from glowstar.service import app as A
    from glowstar.models import registry
    monkeypatch.setattr(registry, "load_current",
                        lambda: (object(), {"version": "20260825T000000",
                                            "trained_at": "2026-08-25T00:00:00"}))
    monkeypatch.setattr(A, "_service", _Loaded("20260701T000000"))

    body = client.get("/health").json()
    assert body["status"] == "degraded"
    joined = " ".join(body.get("warnings", []))
    assert "20260701T000000" in joined and "20260825T000000" in joined, \
        "the alarm must name BOTH versions or an operator cannot act on it"
    assert "restart" in joined.lower(), "the warning must say what to do"
    assert body["model"]["serving_version"] == "20260701T000000"
    assert body["model"]["version"] == "20260825T000000"


def test_health_is_quiet_when_the_worker_is_up_to_date(client, monkeypatch):
    from glowstar.service import app as A
    from glowstar.models import registry
    import datetime as dt
    now = dt.datetime.now().isoformat()
    monkeypatch.setattr(registry, "load_current",
                        lambda: (object(), {"version": "V1", "trained_at": now}))
    monkeypatch.setattr(A, "_service", _Loaded("V1"))

    body = client.get("/health").json()
    joined = " ".join(body.get("warnings", []))
    assert "is promoted" not in joined, \
        f"false alarm on a worker that IS current: {joined}"
    assert body["model"]["serving_version"] == "V1"


def test_health_reports_every_problem_not_just_the_last_one(client, monkeypatch):
    """Each check used to assign out['warning'] directly, so two simultaneous
    faults reported only whichever ran last — the operator fixes what they can
    see and the other stays hidden."""
    from glowstar.service import app as A
    from glowstar.models import registry
    monkeypatch.setattr(registry, "load_current",
                        lambda: (object(), {"version": "NEW",
                                            "trained_at": "2026-01-01T00:00:00"}))
    monkeypatch.setattr(A, "_service", _Loaded("OLD"))
    monkeypatch.setattr(A, "_grid_age_days", lambda: 99)

    body = client.get("/health").json()
    warns = body.get("warnings", [])
    assert len(warns) >= 3, f"expected stale-model + stale-grid + old-model, got {warns}"
    joined = " ".join(warns)
    assert "is promoted" in joined and "grid" in joined and "days old" in joined
    assert body["warning"] == "; ".join(warns), "the legacy string key must still work"
