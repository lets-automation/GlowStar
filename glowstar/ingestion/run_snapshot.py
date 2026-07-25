r"""Recurring snapshot job — run daily (brief Section 3.5).

Pulls the full inventory+sales record set AND the live Master grid, persists each
immutably (dated), and logs schema drift. Designed to be invoked by the OS
scheduler. The repo ships `run_daily_snapshot.bat` and a registered Windows task
(GlowStarSnapshot); to (re)register manually:

  Windows Task Scheduler (daily 02:00, runs if a scheduled start was missed):
    powershell -Command "Register-ScheduledTask -TaskName GlowStarSnapshot ^
      -Action (New-ScheduledTaskAction -Execute 'e:\VS Code\GlowStar\run_daily_snapshot.bat') ^
      -Trigger (New-ScheduledTaskTrigger -Daily -At 2am) ^
      -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable) -Force"
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


def _bank_master_grid() -> None:
    """Best-effort: refresh the live Master grid, bank a dated copy, and append to
    the POINT-IN-TIME grid history.

    Non-fatal: a grid hiccup must never abort the inventory snapshot, which is the
    irreplaceable record.

    The history append is the important half and cannot be backfilled by tomorrow's
    run: `GetCellsHistory` is how we know what a cell read on the day a stone was
    priced, and that is what lets the grid model be TRAINED and, more importantly,
    HONESTLY VALIDATED (today's grid explaining a past sale is leakage — with it the
    grid looks like it beats the engine; point-in-time, it loses). Cell freshness is
    also the single biggest driver of accuracy we measured: a cell edited within 3
    days prices at MAE ~2.0, a 30-day-old one at ~3.1. If this job stops, the grid
    feature rots silently and takes the price with it.
    """
    from .master_grid import refresh_banked_grid, _load_current
    refresh_banked_grid()                       # refresh current.json (latest-wins merge)
    cells = list(_load_current().values())
    if cells:
        gres = save_snapshot(cells, source="master_grid")
        log.info("master-grid snapshot: %s cells, new=%s", gres.n_records, not gres.already_existed)
    try:
        from ..market.grid_history import bank_history
        # A short window each day; the store accumulates. Overlap is idempotent
        # (edits are deduped by timestamp), so a missed day self-heals tomorrow.
        res = bank_history(days=5)
        log.info("grid history: %s cells, %s versions (+%s new)",
                 res["cells"], res["versions"], res["added"])
    except Exception:
        log.exception("Grid-history banking failed (non-fatal) — the grid model will "
                      "train on whatever history is already stored.")


def run(pull_grid_history: bool = False,
        grid_from: str | None = None, grid_to: str | None = None,
        bank_grid: bool = True) -> int:
    """Pull and persist today's snapshots. Returns process exit code.

    Banks (a) the full inventory+sales record set (the irreplaceable series) and,
    best-effort, (b) the live Master grid (`bank_grid`). `pull_grid_history` adds
    a raw GetCellsHistory window snapshot when an explicit range is given.
    """
    try:
        records = channel_partner.get_all_records()
        res = save_snapshot(records, source="channel_partner")
        log.info("inventory snapshot: %s records, new=%s, drift +%s -%s",
                 res.n_records, not res.already_existed, res.added_fields, res.removed_fields)

        if bank_grid:
            try:
                _bank_master_grid()
            except Exception:
                log.exception("Master-grid banking failed (non-fatal); inventory snapshot kept.")

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
