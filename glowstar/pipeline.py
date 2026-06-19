"""End-to-end pipeline: ingest -> market artifacts -> train -> serve.

Runs fully end to end with whatever data is available — no live-credential
blocker. Record ingestion uses a source abstraction:

  * LIVE   — pull from the Channel Partner API (when credentials are set) and
             bank an immutable snapshot.
  * FILE   — the latest banked snapshot, else the shipped records.json.

So `build_service()` always returns a working, trained PricingService. When
credentials are provided it transparently switches to the live pull.

Run:  python -m glowstar.pipeline            # build + smoke-price a sample stone
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .config import ARTIFACTS_DIR, PATHS
from .data.loaders import load_records, sold_stones
from .ingestion.http import CredentialError
from .ingestion.snapshots import SNAPSHOT_ROOT, save_snapshot
from .models.engine import PricingEngine, EngineConfig
from .service.pricing_service import PricingService
from .feedback import store as fbstore

log = logging.getLogger(__name__)


def ingest_records(prefer_live: bool = False) -> Path:
    """Return a path to a records JSON to load, pulling live if asked & possible.

    LIVE path banks a snapshot and returns it; otherwise the latest snapshot,
    otherwise the shipped records.json. Never fails for lack of credentials.
    """
    if prefer_live:
        try:
            from .ingestion import channel_partner
            records = channel_partner.get_all_records()
            res = save_snapshot(records, source="channel_partner")
            log.info("Live pull banked: %s", res.path)
            return res.path
        except CredentialError as e:
            log.warning("Live pull unavailable (%s); falling back to file source.", e)
        except Exception:
            log.exception("Live pull failed; falling back to file source.")

    snap_dir = SNAPSHOT_ROOT / "channel_partner"
    if snap_dir.exists():
        snaps = sorted(snap_dir.glob("*.json"))
        if snaps:
            log.info("Using latest banked snapshot: %s", snaps[-1].name)
            return snaps[-1]
    log.info("Using shipped records.json")
    return PATHS.records


def ensure_market_artifacts() -> bool:
    """True if the market anchor/BGM artifacts exist. Warns (does not crash) if
    not — the 6.2GB aggregation (python -m glowstar.market.aggregate_bulk) builds
    them; the engine still runs without, just without the market anchor."""
    seg = ARTIFACTS_DIR / "market_segments.json"
    bgm = ARTIFACTS_DIR / "bgm_discounts.json"
    ok = seg.exists() and bgm.exists()
    if not ok:
        log.warning("Market artifacts missing in %s — run "
                    "`python -m glowstar.market.aggregate_bulk` to enable the "
                    "market anchor + BGM deductions.", ARTIFACTS_DIR)
    return ok


def build_service(prefer_live: bool = False, config: EngineConfig | None = None) -> PricingService:
    """Ingest, ensure artifacts, train on all sold history + feedback, and serve."""
    records_path = ingest_records(prefer_live)
    ensure_market_artifacts()
    df, rep = load_records(records_path)
    log.info(rep.summary())
    sold = sold_stones(df, drop_outliers=True)
    feedback = fbstore.load_all()
    engine = PricingEngine(config or EngineConfig()).fit(sold, feedback_records=feedback)
    log.info("Engine trained on %s sold stones + %s feedback records.",
             f"{len(sold):,}", len(feedback))
    return PricingService(engine=engine)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .service.pricing_service import StoneIn
    svc = build_service(prefer_live=True)        # tries live, falls back to file
    demo = StoneIn(StoneId="PIPELINE-DEMO", Shape_full="Round", Weight=1.01,
                   Color="G", Clarity="VS2", CPS="3EX", Lab="GIA", Rap=8200)
    out = svc.price(demo)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
