"""Durable store for everything the live service produces.

WHY THIS EXISTS
---------------
Until now the engine wrote to files: decisions to an append-only JSONL, quotes
nowhere at all. That is fine for a script one person runs. It is not fine once
the client's CRM is calling us all day:

  * CONCURRENT WRITES. An append-only text file has no transaction. The nightly
    job and a live API request writing at the same moment is a class of bug you
    do not want to debug at 3 a.m. on someone else's production floor.
  * NO QUOTE HISTORY. We could not answer "what did we quote for this stone in
    March, and which model said it?" — which is the first question asked when a
    price is disputed, and the MOU's audit requirement.
  * NO SCORE HISTORY. The AI-score weights are deliberately unfitted (see
    ai_score.py). Refitting them later needs the scores we gave AND what the
    stone eventually did. Without storing them, that evidence never accumulates.

ONE CODEBASE, TWO ENGINES
-------------------------
`GS_DATABASE_URL` decides:
    unset                      -> SQLite at data/glowstar.db  (local, dev, tests)
    postgresql+psycopg://...   -> PostgreSQL                  (production)
Same schema, same code. Nothing to port on deployment day — the only difference
between a laptop and the production server is one environment variable.

WRITES MUST NEVER BREAK A PRICE
-------------------------------
Every write here is best-effort and swallowed on failure. If the database is
down, the desk still gets its price; we lose an audit row, which is recoverable,
instead of the quote, which is not. Failures are logged loudly.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, MetaData,
                        String, Table, Text, create_engine, insert, select)
from sqlalchemy.engine import Engine

from ..config import DATA_DIR

log = logging.getLogger(__name__)

metadata = MetaData()

# Every price we have ever published. The audit trail: which stone, what we said,
# which model version said it. Never updated, only inserted.
quotes = Table(
    "quotes", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("stone_id", String(64), index=True),
    Column("certificate_no", String(64), index=True),
    Column("shape", String(32)), Column("weight", Float),
    Column("color", String(8)), Column("clarity", String(8)),
    Column("cps", String(16)), Column("fluorescence", String(16)),
    Column("rap", Float),
    Column("discount", Float),            # what we quoted
    Column("ppc", Float), Column("net", Float),
    Column("ci_low", Float), Column("ci_high", Float),
    Column("method", String(32)),
    Column("comparable_count", Integer),
    Column("flags", Text),                # JSON list
    Column("model_version", String(32), index=True),
    Column("source", String(32)),         # frontoffice | api | excel
)

# The desk's decision on a quote. Mirrors the JSONL record so the existing
# feedback pipeline keeps working unchanged while this becomes the system of
# record.
decisions = Table(
    "decisions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("stone_id", String(64), index=True),
    Column("certificate_no", String(64), index=True),
    Column("decision", String(16), index=True),
    Column("suggested_discount", Float),
    Column("human_discount", Float),
    Column("variance_pts", Float),
    Column("needs_attention", Boolean),
    Column("reason_code", String(48), index=True),
    Column("note", Text),
    Column("user", String(64)),
    Column("shape", String(32)), Column("weight", Float),
    Column("color", String(8)), Column("clarity", String(8)),
    Column("trainable", Boolean),         # False when no desk price was supplied
)

# The AI-score components we published, so the weights can be refit later against
# what the stone actually did.
scores = Table(
    "scores", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("stone_id", String(64), index=True),
    Column("competition", Integer), Column("liquidity", Integer),
    Column("price_competitive", Integer), Column("turnaround", Integer),
    Column("market_strength", Integer), Column("urgency", Integer),
    Column("confidence", Integer), Column("final_score", Integer),
    Column("tradeability", String(16)),
    Column("tradeability_days", Integer),
    Column("model_version", String(32)),
)

_engine: Engine | None = None


def database_url() -> str:
    """Production Postgres when configured, else a local SQLite file."""
    url = os.environ.get("GS_DATABASE_URL", "").strip()
    if url:
        return url
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(DATA_DIR / 'glowstar.db').as_posix()}"


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = database_url()
        kw: dict = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            # WAL lets a reader run while a writer commits — the nightly job and a
            # live request no longer block each other on the dev/SQLite path.
            kw["connect_args"] = {"timeout": 30, "check_same_thread": False}
        _engine = create_engine(url, **kw)
        if url.startswith("sqlite"):
            with _engine.begin() as c:
                from sqlalchemy import text
                c.execute(text("PRAGMA journal_mode=WAL"))
                c.execute(text("PRAGMA synchronous=NORMAL"))
        metadata.create_all(_engine)      # idempotent: safe on every boot
        log.info("Store ready: %s", url.split("@")[-1])   # never log credentials
    return _engine


def _execute(build_stmt) -> bool:
    """Best-effort write. A store failure must never cost the desk a price.

    `build_stmt` is a zero-arg callable returning the statement, so building it
    is inside the try as well — a malformed row must fail the same soft way a
    dead database does.

    This was a @contextmanager yielding the connection. That version was NOT
    best-effort: when `get_engine()` itself raised (database down), the generator
    exited without ever yielding and @contextmanager turned that into
    `RuntimeError: generator didn't yield`, which propagated into the caller and
    would have taken pricing down precisely when the store was already broken.
    A plain function has no such failure mode.
    """
    try:
        with get_engine().begin() as conn:
            conn.execute(build_stmt())
        return True
    except Exception:
        log.exception("store write failed — the price was still served")
        return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record_quote(*, facts: dict, stone: dict, model_version: str | None,
                 source: str = "api") -> None:
    """Persist a published price. Called on every quote; never raises."""
    _execute(lambda: insert(quotes).values(
        ts=_now(),
        stone_id=str(stone.get("StoneId") or stone.get("stoneId") or "")[:64],
        certificate_no=str(stone.get("CertificateNo") or
                           stone.get("certificateNo") or "")[:64],
        shape=str(stone.get("Shape_full") or stone.get("shape") or "")[:32],
        weight=_f(stone.get("Weight") or stone.get("weight")),
        color=str(stone.get("Color") or stone.get("color") or "")[:8],
        clarity=str(stone.get("Clarity") or stone.get("clarity") or "")[:8],
        cps=str(stone.get("CPS") or stone.get("cps") or "")[:16],
        fluorescence=str(stone.get("Fluorescence") or
                         stone.get("fluorescence") or "")[:16],
        rap=_f(stone.get("Rap") or stone.get("rap")),
        discount=_f(facts.get("suggested_discount")),
        ppc=_f(facts.get("suggested_ppc")), net=_f(facts.get("suggested_net")),
        ci_low=_f(facts.get("ci_discount_low")),
        ci_high=_f(facts.get("ci_discount_high")),
        method=str(facts.get("method") or "")[:32],
        comparable_count=_i(facts.get("comparable_count")),
        flags=json.dumps(facts.get("flags") or []),
        model_version=(model_version or "")[:32],
        source=source[:32],
    ))


def record_decision(*, rec: dict, variance: float | None,
                    needs_attention: bool, trainable: bool = True) -> None:
    """Persist a desk decision alongside the JSONL log."""
    _execute(lambda: insert(decisions).values(
        ts=_now(), stone_id=str(rec.get("stone_id") or "")[:64],
        certificate_no=str(rec.get("certificate_no") or "")[:64],
        decision=str(rec.get("decision") or "")[:16],
        suggested_discount=_f(rec.get("suggested_discount")),
        human_discount=_f(rec.get("human_discount")),
        variance_pts=_f(variance), needs_attention=bool(needs_attention),
        reason_code=str(rec.get("reason_code") or "")[:48] or None,
        note=str(rec.get("note") or ""), user=str(rec.get("user") or "")[:64],
        shape=str(rec.get("shape_full") or "")[:32], weight=_f(rec.get("weight")),
        color=str(rec.get("color") or "")[:8],
        clarity=str(rec.get("clarity") or "")[:8],
        trainable=bool(trainable),
    ))


def record_scores(*, stone_id: str, s: dict, tradeability: dict | None,
                  model_version: str | None) -> None:
    """Persist the AI-score components, so the weights can be refit later."""
    _execute(lambda: insert(scores).values(
        ts=_now(), stone_id=str(stone_id or "")[:64],
        competition=_i(s.get("CompetitionScore")),
        liquidity=_i(s.get("LiquidityScore")),
        price_competitive=_i(s.get("PriceCompetitiveScore")),
        turnaround=_i(s.get("TurnaroundScore")),
        market_strength=_i(s.get("MarketStrengthScore")),
        urgency=_i(s.get("UrgencyScore")),
        confidence=_i(s.get("ConfidenceScore")),
        final_score=_i(s.get("FinalAIScore")),
        tradeability=str((tradeability or {}).get("label") or "")[:16] or None,
        tradeability_days=_i((tradeability or {}).get("median_days")),
        model_version=(model_version or "")[:32],
    ))


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def counts() -> dict:
    """Row counts — surfaced by /health and `status` so an empty store is visible."""
    try:
        from sqlalchemy import func
        with get_engine().connect() as c:
            return {t.name: int(c.execute(select(func.count()).select_from(t)).scalar() or 0)
                    for t in (quotes, decisions, scores)}
    except Exception as e:
        return {"error": type(e).__name__}
