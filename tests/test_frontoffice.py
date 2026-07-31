"""The FrontOffice contract (client spec, 29-07-2026).

Their CRM binds to these field names and this response shape, so a rename here is
a breaking change on their side. These tests pin the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient          # noqa: E402

from glowstar.service import app as app_mod        # noqa: E402
from glowstar.service import frontoffice as fo     # noqa: E402
from glowstar.service import tradeability as tr    # noqa: E402


FO_STONE = {
    "stoneId": "S26-1", "certificateNo": "12345", "shape": "Round", "weight": 0.52,
    "color": "G", "clarity": "VS1", "cut": "EX", "polish": "EX", "symmetry": "EX",
    "fluorescence": "Non", "lab": "GIA",
    "inclusion": {"brown": "NO", "milky": "LML", "shade": "NO", "green": "NO",
                  "eyeClean": "YES", "luster": "EX", "bowtie": "NO"},
    "measurement": {"length": 5.18, "width": 5.21, "depth": 3.19, "ratio": 1.01,
                    "table": 57.0, "mGrade": "A1"},
}


class _FakeService:
    class _Eng:
        def set_corrections(self, *a, **k): pass
    engine = _Eng()

    def price(self, stone):
        return {"suggestion": {
            "stone_id": stone.StoneId, "suggested_discount": -43.78,
            "suggested_ppc": 1068.14, "suggested_net": 555.43,
            "ci_discount_low": -47.30, "ci_discount_high": -38.97,
            "comparable_count": 1763, "method": "model+anchor", "flags": [],
        }, "market": {}, "explanation": {"text": "1,763 comparable stones."}}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "_service", _FakeService())
    import glowstar.feedback.store as store
    monkeypatch.setattr(store, "FEEDBACK_LOG", tmp_path / "d.jsonl")
    monkeypatch.delenv("GS_API_KEY", raising=False)
    # keep the heavy days-to-sell table out of API tests
    monkeypatch.setattr(tr, "tradeability_for",
                        lambda *a, **k: {"label": "High", "median_days": 35,
                                         "basis": "segment Round|G|VS1"})
    return TestClient(app_mod.app)


# --------------------------------------------------------------------------
# Spec #1 — bulk pricing
# --------------------------------------------------------------------------
def test_response_carries_every_field_their_spec_asks_for(client):
    r = client.post("/frontoffice/price", json=[FO_STONE])
    assert r.status_code == 200
    row = r.json()[0]
    for field in ("StoneId", "CertificateNo", "AIDiscount", "Reason",
                  "Tradeability", "AIScore", "ConfidenceScore"):
        assert field in row, f"their spec requires {field}"
    assert row["StoneId"] == "S26-1" and row["CertificateNo"] == "12345"
    assert row["AIDiscount"] == -43.78
    assert row["Tradeability"] == "High"
    assert isinstance(row["ConfidenceScore"], int)


def test_fields_we_cannot_price_on_are_named_not_silently_dropped(client):
    """eyeClean/luster/bowtie are in their request but not in the sales history,
    so the model has never seen them. Accepting them silently would imply they
    influenced the price."""
    row = client.post("/frontoffice/price", json=[FO_STONE]).json()[0]
    unused = row.get("ReceivedNotYetPriced", [])
    assert "eyeClean" in unused and "luster" in unused and "bowtie" in unused
    # ...but the ones we DO use must not be listed as unused
    assert "brown" not in unused and "milky" not in unused


def test_ai_score_is_computed_without_demand(client):
    """Client removed Demand (2026-07-30): the search feed has no buyer identity,
    offers or video views. The score is built from the six measurable components."""
    row = client.post("/frontoffice/price", json=[FO_STONE]).json()[0]
    for k in ("CompetitionScore", "LiquidityScore", "PriceCompetitiveScore",
              "TurnaroundScore", "MarketStrengthScore", "UrgencyScore",
              "ConfidenceScore", "FinalAIScore"):
        assert k in row, f"AI score must expose {k}"
    assert row["AIScore"] == row["FinalAIScore"]
    assert row["DemandScore"] is None
    assert "removed at client request" in row["DemandScoreStatus"]
    # every score carries its reason — a bare number the desk cannot interrogate
    # is a number they stop trusting
    assert row["ScoreBasis"]["Competition"]


def test_scores_are_all_on_a_0_100_scale_where_high_is_good():
    from glowstar.service import ai_score
    r = ai_score.compute(our_discount=-47.7, market_discount=-44.0,
                         market_depth=1763, own_sales=210, median_days=28,
                         age_days=35, confidence=77)
    for k, v in r.items():
        if k.endswith("Score") and isinstance(v, int):
            assert 0 <= v <= 100, f"{k}={v} is off the 0-100 scale"
    # crowded market must score LOWER than an empty one
    crowded = ai_score.competition_score(50_000)[0]
    quiet = ai_score.competition_score(20)[0]
    assert quiet > crowded
    # cheaper than market scores HIGHER than dearer
    cheap = ai_score.price_competitive_score(-50.0, -45.0)[0]
    dear = ai_score.price_competitive_score(-40.0, -45.0)[0]
    assert cheap > dear
    # an overdue stone scores LOW on urgency (low = act now)
    assert ai_score.urgency_score(200, 30)[0] < ai_score.urgency_score(20, 30)[0]


def test_final_score_renormalises_when_data_is_missing():
    """A stone with thin data must be scored on what IS known, not penalised for
    data it was never going to have — and never given a default dressed up as
    an assessment."""
    from glowstar.service import ai_score
    partial = ai_score.final_score({"Competition": 80, "Liquidity": 60,
                                    "PriceCompetitive": None, "Turnaround": None,
                                    "MarketStrength": None, "Urgency": None})
    assert partial[0] is not None and "missing" in partial[1]
    assert ai_score.final_score({k: None for k in ai_score.WEIGHTS}) == (
        None, "no component could be computed")


def test_weights_sum_to_one():
    from glowstar.service import ai_score
    assert abs(sum(ai_score.WEIGHTS.values()) - 1.0) < 1e-9


def test_one_malformed_stone_never_fails_the_book(client):
    bad = {"stoneId": "BAD", "shape": "Round"}          # missing weight/color/clarity
    out = client.post("/frontoffice/price", json=[FO_STONE, bad]).json()
    assert len(out) == 2
    ok = [r for r in out if r["StoneId"] == "S26-1"][0]
    broke = [r for r in out if r["StoneId"] == "BAD"][0]
    assert ok["AIDiscount"] == -43.78                   # the good one still priced
    assert broke["AIDiscount"] is None and broke["Error"]


def test_cps_is_derived_when_not_supplied():
    """The model learned cut from the client's clean vocabulary; an unseen
    combined code loses the cut signal entirely."""
    s = fo.to_stone_in(fo.FrontOfficeStone(**{**FO_STONE, "cps": None}))
    assert s.CPS == "3EX"                                # EX/EX/EX -> 3EX
    s2 = fo.to_stone_in(fo.FrontOfficeStone(**{**FO_STONE, "cps": None,
                                               "cut": "VG", "polish": "EX",
                                               "symmetry": "EX"}))
    assert s2.CPS == "VG"


def test_tinge_and_measurements_reach_the_engine():
    s = fo.to_stone_in(fo.FrontOfficeStone(**FO_STONE))
    assert s.Milky == "LML" and s.Brown == "NO"
    assert s.Length == 5.18 and s.Width == 5.21


# --------------------------------------------------------------------------
# Spec #2 — the desk's reason
# --------------------------------------------------------------------------
def test_reason_without_the_desks_price_is_stored_but_NOT_trainable(client):
    """Their doc sends certificateNo + reason + aiDiscount and no desk price.
    That records THAT we were wrong, never WHAT right looks like."""
    r = client.post("/frontoffice/reason", json={
        "certificateNo": "12345", "reason": "market_moved", "aiDiscount": -43.78})
    assert r.status_code == 200
    b = r.json()
    assert b["recorded"] is True
    assert b["trainable"] is False
    assert "deskDiscount" in b["note"]


def test_reason_with_the_desks_price_IS_trainable(client):
    r = client.post("/frontoffice/reason", json={
        "certificateNo": "12345", "reason": "market_moved",
        "aiDiscount": -43.78, "deskDiscount": -48.0})
    b = r.json()
    assert b["trainable"] is True
    assert b["variance_pts"] == pytest.approx(4.22, abs=0.01)
    assert b["needs_attention"] is True          # 4.2 pts > the 2-pt threshold


# --------------------------------------------------------------------------
# Spec #3 — master grid cell
# --------------------------------------------------------------------------
def test_master_discount_prices_a_weight_RANGE(client):
    r = client.post("/frontoffice/master-discount", json={
        "fromWeight": 0.30, "toWeight": 0.39, "color": "G", "clarity": "VS1",
        "cps": "3EX", "floro": "Non"})
    assert r.status_code == 200
    b = r.json()
    assert b["fromWeight"] == 0.30 and b["toWeight"] == 0.39
    assert b["pricedAtWeight"] == pytest.approx(0.34, abs=0.01)   # midpoint, 2dp
    assert b["AIDiscount"] == -43.78
    assert b["MarketComparables"] == 1763


def test_master_discount_rejects_a_backwards_range(client):
    r = client.post("/frontoffice/master-discount", json={
        "fromWeight": 0.90, "toWeight": 0.30, "color": "G", "clarity": "VS1"})
    assert r.status_code == 422


# --------------------------------------------------------------------------
# Tradeability — the censoring correction is the whole point
# --------------------------------------------------------------------------
def test_unsold_stock_is_censored_not_ignored():
    """Measuring only SOLD stones biases days-to-sell fast: the slowest goods have
    not sold yet, so they vanish from a sold-only average. On the real book the
    correction moves the median 42 -> 46 days.

    (The larger 75-day figure that censoring alone produces is itself wrong: it
    ignores LEFT-TRUNCATION — stock keeps pre-window survivors while their
    contemporaries that already sold are absent from the records. Both biases
    must be corrected; see tradeability.py.)"""
    # 6 quick sales; 14 stones still sitting unsold at day 500.
    # Sold-only says "median 10 days". But only 6 of 20 have actually sold by
    # then, so the true median is NOT reached at day 10 — the fast answer is an
    # artefact of throwing the unsold stock away.
    dur = np.array([10.0] * 6 + [500.0] * 14)
    obs = np.array([True] * 6 + [False] * 14)
    sold_only = float(np.median(dur[obs]))
    km = tr._km_median(dur, obs)
    assert sold_only == 10.0
    assert km > sold_only, "censored stones must drag the estimate slower"

    # And where half HAVE sold by day 10, 10 days is the honest median.
    assert tr._km_median(np.array([10.0] * 10 + [500.0] * 10),
                         np.array([True] * 10 + [False] * 10)) == 10.0


def test_median_not_reached_returns_infinity_not_a_guess():
    """If most stock has not sold, the honest answer is 'we don't know yet'."""
    dur = np.array([100.0] * 9 + [5.0])
    obs = np.array([False] * 9 + [True])
    assert tr._km_median(dur, obs) == float("inf")


def test_labels_are_the_five_the_client_asked_for():
    assert tr.LABELS == ("High", "Semi High", "Medium", "Semi Slow", "Slow")
    cut = (35, 51, 69, 94)
    assert tr.label_for_days(20, cut) == "High"
    assert tr.label_for_days(60, cut) == "Medium"
    assert tr.label_for_days(200, cut) == "Slow"


def test_confidence_score_reflects_real_reliability():
    tight = {"ci_discount_low": -46, "ci_discount_high": -40,
             "comparable_count": 500, "flags": [], "method": "model+anchor"}
    wide = {"ci_discount_low": -60, "ci_discount_high": -35,
            "comparable_count": 3, "flags": ["rare_shape"], "method": "fallback"}
    hi, lo = fo._confidence_score(tight), fo._confidence_score(wide)
    assert hi > 70 and lo < 30 and 0 <= lo <= 100 and 0 <= hi <= 100
