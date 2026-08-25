"""The durable store — quotes, decisions and scores must actually land.

Both of these were REAL bugs caught by querying the database instead of trusting
a 200 response: the decision mirror silently never ran, and every quote row was
written with an empty model_version (an audit row that cannot name the model that
produced it is not an audit row). A green API response proves nothing about
persistence, so these tests check the rows.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import desc, select                     # noqa: E402

from glowstar.store import db                           # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A throwaway SQLite database per test."""
    monkeypatch.setenv("GS_DATABASE_URL", f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    monkeypatch.setattr(db, "_engine", None)
    db.get_engine()
    yield db
    monkeypatch.setattr(db, "_engine", None)


def test_sqlite_by_default_postgres_when_configured(monkeypatch):
    """One codebase, two engines — nothing to port on deployment day."""
    monkeypatch.delenv("GS_DATABASE_URL", raising=False)
    assert db.database_url().startswith("sqlite:///")
    monkeypatch.setenv("GS_DATABASE_URL", "postgresql+psycopg://u:p@host/db")
    assert db.database_url().startswith("postgresql+psycopg://")


def test_quote_is_recorded_with_the_model_that_made_it(store):
    store.record_quote(
        facts={"suggested_discount": -47.6, "suggested_ppc": 996.2,
               "suggested_net": 518.0, "ci_discount_low": -51.1,
               "ci_discount_high": -42.8, "method": "model+anchor",
               "comparable_count": 1763, "flags": ["x"]},
        stone={"StoneId": "S1", "Shape_full": "Round", "Weight": 0.52,
               "Color": "G", "Clarity": "VS1", "Rap": 2100},
        model_version="20260728T102722", source="frontoffice")
    with store.get_engine().connect() as c:
        r = c.execute(select(store.quotes).order_by(desc(store.quotes.c.id))).first()
    assert r.stone_id == "S1" and r.discount == -47.6
    assert r.model_version == "20260728T102722", "an audit row must name its model"
    assert r.source == "frontoffice"


def test_decision_is_mirrored_from_the_feedback_log(store, tmp_path, monkeypatch):
    """The JSONL append and the database row must both happen. The mirror once
    silently did not run at all — the API still returned 200."""
    from glowstar.feedback.store import Decision, FeedbackRecord, record

    before = store.counts()["decisions"]
    rec = FeedbackRecord(
        stone_id="S2", decision=Decision.OVERRIDE.value, suggested_discount=-47.5,
        suggested_net=500.0, shape_full="Round", weight=0.52, color="G",
        clarity="VS1", human_discount=-49.0, user="milan")
    record(rec, path=tmp_path / "d.jsonl")

    assert store.counts()["decisions"] == before + 1
    with store.get_engine().connect() as c:
        r = c.execute(select(store.decisions).order_by(desc(store.decisions.c.id))).first()
    assert r.stone_id == "S2" and r.human_discount == -49.0
    assert r.variance_pts == pytest.approx(1.5)
    assert r.needs_attention is False        # 1.5 pts is inside the 2-pt threshold
    assert r.trainable is True               # a desk price was supplied


def test_a_reason_with_no_desk_price_is_marked_not_trainable(store, tmp_path):
    from glowstar.feedback.store import Decision, FeedbackRecord, record
    rec = FeedbackRecord(
        stone_id="S3", decision=Decision.REJECT.value, suggested_discount=-47.5,
        suggested_net=500.0, shape_full="Round", weight=0.52, color="G",
        clarity="VS1", reason_code="market_moved")
    record(rec, path=tmp_path / "d.jsonl")
    with store.get_engine().connect() as c:
        r = c.execute(select(store.decisions).order_by(desc(store.decisions.c.id))).first()
    assert r.trainable is False, "no desk price means nothing to learn from"


def test_scores_are_kept_so_the_weights_can_be_refit_later(store):
    store.record_scores(stone_id="S4",
                        s={"CompetitionScore": 29, "LiquidityScore": 85,
                           "PriceCompetitiveScore": 71, "TurnaroundScore": 84,
                           "MarketStrengthScore": 71, "UrgencyScore": 82,
                           "ConfidenceScore": 77, "FinalAIScore": 71},
                        tradeability={"label": "High", "median_days": 28},
                        model_version="v1")
    with store.get_engine().connect() as c:
        r = c.execute(select(store.scores).order_by(desc(store.scores.c.id))).first()
    assert r.final_score == 71 and r.tradeability == "High"
    assert r.competition == 29 and r.urgency == 82


def test_a_store_failure_never_costs_the_desk_a_price(store, monkeypatch):
    """Best-effort by design: if the database is down we lose an audit row, which
    is recoverable — not the quote, which is not."""
    def _boom():
        raise RuntimeError("database is down")
    monkeypatch.setattr(db, "get_engine", _boom)
    store.record_quote(facts={"suggested_discount": -40.0}, stone={"StoneId": "S5"},
                       model_version="v1")          # must not raise
    store.record_scores(stone_id="S5", s={}, tradeability=None, model_version="v1")
    assert "error" in db.counts()


def test_create_all_is_idempotent(store):
    """Every boot calls it; a second call must not fail or wipe anything."""
    store.record_quote(facts={"suggested_discount": -40.0}, stone={"StoneId": "S6"},
                       model_version="v1")
    store.metadata.create_all(store.get_engine())
    assert store.counts()["quotes"] == 1


# --- rate limiting ----------------------------------------------------------
# The client declined an IP allowlist, so the API key is the only fence. The
# realistic threat is a retry loop on their side saturating the box and starving
# the 02:30 retrain — an availability bug that turns into an accuracy bug.

def test_rate_limit_allows_then_blocks_then_recovers(monkeypatch):
    from glowstar.service import ratelimit
    monkeypatch.setenv("GS_RATE_LIMIT", "5")
    monkeypatch.setenv("GS_RATE_WINDOW_S", "60")
    ratelimit.reset()

    for i in range(5):
        ok, remaining, _ = ratelimit.check("caller-a", now=1000.0)
        assert ok, f"request {i} should be allowed"
        assert remaining == 4 - i

    ok, _, retry_after = ratelimit.check("caller-a", now=1000.0)
    assert not ok, "the 6th request inside the window must be rejected"
    assert 0 < retry_after <= 60

    # The window slides — it is not a fixed bucket that resets on the minute.
    ok, _, _ = ratelimit.check("caller-a", now=1061.0)
    assert ok, "a slot must free up once the oldest hit ages out"


def test_rate_limit_is_per_caller(monkeypatch):
    """One runaway CRM instance must not starve the desk's other machines."""
    from glowstar.service import ratelimit
    monkeypatch.setenv("GS_RATE_LIMIT", "2")
    ratelimit.reset()
    assert ratelimit.check("a", now=1.0)[0]
    assert ratelimit.check("a", now=1.0)[0]
    assert not ratelimit.check("a", now=1.0)[0]
    assert ratelimit.check("b", now=1.0)[0], "a different caller has its own budget"


def test_rate_limit_can_be_disabled(monkeypatch):
    """An operator must be able to switch it off for a bulk migration."""
    from glowstar.service import ratelimit
    monkeypatch.setenv("GS_RATE_LIMIT", "0")
    ratelimit.reset()
    for _ in range(500):
        assert ratelimit.check("anyone", now=1.0)[0]


def test_rate_limit_tracking_dict_is_bounded(monkeypatch):
    """Forged keys must not grow the counter dict without bound."""
    from glowstar.service import ratelimit
    monkeypatch.setenv("GS_RATE_LIMIT", "10")
    ratelimit.reset()
    for i in range(ratelimit._MAX_TRACKED + 200):
        ratelimit.check(f"key-{i}", now=float(i))
    assert len(ratelimit._hits) <= ratelimit._MAX_TRACKED


# --- audit trail on every published price ----------------------------------
# Only /frontoffice/price recorded quotes. /price and /price/batch published
# prices that left NO record, so anything priced through them was invisible
# afterwards — and "what did you quote in March, and why?" is the first
# question asked when a price is disputed (MOU audit requirement).

def test_price_endpoints_are_wired_to_the_audit_trail(monkeypatch):
    """BEHAVIOURAL. The first version of this test grepped app.py for the exact
    call text `_audit(svc.price(s), s, "api-batch")`. That is not a test of the
    audit trail, it is a test of one spelling: batching /price/batch into a
    single `price_many` call broke it while auditing still worked perfectly,
    and — far worse — it would have passed just as happily if `_audit` had been
    deleted and replaced by a differently-named no-op. Assert that a ROW LANDS.
    """
    from glowstar.service import app as A
    from glowstar.store import db

    recorded: list[dict] = []
    monkeypatch.setattr(db, "record_quote",
                        lambda **kw: recorded.append(kw) or 1)
    monkeypatch.delenv("GS_API_KEY", raising=False)

    stone = {"StoneId": "AUDIT-1", "Shape_full": "Round", "Weight": 0.9,
             "Color": "G", "Clarity": "VS1", "CPS": "3EX", "Fluorescence": "Non"}
    c = TestClient(A.app)

    assert c.post("/price", json=stone).status_code == 200
    assert any(r.get("stone", {}).get("StoneId") == "AUDIT-1" for r in recorded),         "/price published a price that left no audit row"

    n_before = len(recorded)
    r = c.post("/price/batch", json=[stone, dict(stone, StoneId="AUDIT-2")])
    assert r.status_code == 200
    ids = {rec.get("stone", {}).get("StoneId") for rec in recorded[n_before:]}
    assert {"AUDIT-1", "AUDIT-2"} <= ids,         f"/price/batch published prices that left no audit row (saw {ids})"


def test_audit_failure_never_costs_the_price(monkeypatch):
    """A dead store must lose an audit row, never the quote."""
    from glowstar.store import db
    monkeypatch.setattr(db, "get_engine",
                        lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    # record_quote swallows everything internally; it must not raise.
    db.record_quote(facts={"suggested_discount": -50.0}, stone={"StoneId": "X"},
                    model_version="v1", source="api")


# --- one definition of "trainable", persisted as a boolean ------------------
# Two copies of this rule drifted apart in production: the API reported
# trainable=true whenever a desk price was present, and the DB column did the
# same — so 14 real desk corrections are stored marked trainable when not one of
# them carries a stone and none can ever train a price cell.

def _fb(**kw):
    from glowstar.feedback.store import FeedbackRecord
    base = dict(stone_id="S1", decision="override", suggested_discount=-50.0,
                suggested_net=0.0, shape_full="Round", weight=1.01,
                color="G", clarity="VS1", human_discount=-55.0)
    base.update(kw)
    return FeedbackRecord(**base)


def test_trainable_needs_a_desk_price():
    from glowstar.feedback.store import is_trainable
    assert is_trainable(_fb()) is True
    assert is_trainable(_fb(human_discount=None)) is False


def test_trainable_needs_a_real_stone():
    """The production bug: desk price present, stone hardcoded to NA/0."""
    from glowstar.feedback.store import is_trainable
    assert is_trainable(_fb(shape_full="NA", weight=0.0,
                            color="NA", clarity="NA")) is False
    assert is_trainable(_fb(weight=0.0)) is False
    assert is_trainable(_fb(shape_full="NA")) is False
    assert is_trainable(_fb(color="NA")) is False
    assert is_trainable(_fb(clarity="NA")) is False


def test_trainable_survives_junk_weight():
    from glowstar.feedback.store import is_trainable
    assert is_trainable(_fb(weight=None)) is False
    assert is_trainable(_fb(weight="abc")) is False


def test_api_and_store_share_one_definition():
    """Two copies of the rule is exactly how they drifted apart."""
    import inspect
    from glowstar.service import app as A
    src = inspect.getsource(A)
    assert "is_trainable(rec)" in src, "API must use the shared definition"
    assert '"trainable": desk is not None' not in src, "the old duplicate rule is back"
