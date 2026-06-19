"""Recurring snapshot job — run daily (brief Section 3.5).

Pulls the full inventory+sales record set and (optionally) a Diamanto grid-
history window, persists each immutably, and logs schema drift. Designed to be
invoked by the OS scheduler:

  Windows Task Scheduler (daily 02:00):
    schtasks /Create /SC DAILY /ST 02:00 /TN GlowStarSnapshot ^
      /TR ".venv\Scripts\python.exe -m glowstar.ingestion.run_snapshot"
  cron:
    0 2 * * *  cd /path/to/glowstar && .venv/bin/python -m glowstar.ingestion.run_snapshot

This is cheap now and impossible to recover later: every day banked grows the
series the velocity and market-trend models need.
"""

from __future__ import annotations

import logging
import sys

from . import channel_partner, diamanto
from .http import CredentialError
from .snapshots import save_snapshot

log = logging.getLogger(__name__)


def run(pull_grid_history: bool = False,
        grid_from: str | None = None, grid_to: str | None = None) -> int:
    """Pull and persist today's snapshots. Returns process exit code."""
    try:
        records = channel_partner.get_all_records()
        res = save_snapshot(records, source="channel_partner")
        log.info("inventory snapshot: %s records, new=%s, drift +%s -%s",
                 res.n_records, not res.already_existed, res.added_fields, res.removed_fields)

        if pull_grid_history and grid_from and grid_to:
            cells = diamanto.get_cells_history(grid_from, grid_to)
            gres = save_snapshot(cells, source="diamanto_grid")
            log.info("grid-history snapshot: %s cells", gres.n_records)
        return 0
    except CredentialError as e:
        log.error("Snapshot aborted — %s", e)
        return 2
    except Exception:
        log.exception("Snapshot job failed")
        return 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run())


if __name__ == "__main__":
    main()
