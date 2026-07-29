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
