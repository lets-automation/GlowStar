"""Request rate limiting for the public pricing API.

WHY THIS EXISTS
---------------
The client declined an IP allowlist — their CRM calls us "from anywhere", on a
public address. That is a legitimate choice, but it removes the outer fence: the
API key becomes the ONLY thing between the internet and a service that holds a
~1 GB model in memory and shares a box with the nightly retrain.

The realistic threat here is not an attacker. It is a **misconfigured retry loop
on their side** — a CRM screen that re-requests on every render, or a failed
batch retried in a `while` loop. That saturates two uvicorn workers, and because
the nightly retrain runs on the same machine, a daytime flood can starve the
02:30 rebuild. Stale grid data costs ~1 point of accuracy per week of staleness
(CLAUDE.md Trap 8), so an availability problem becomes an accuracy problem.

DESIGN — deliberately boring
----------------------------
A sliding window counter held in process memory. No Redis, no extra service to
deploy, patch or monitor: this must not add a moving part to a deployment the
client's team has to run without us.

Consequences of in-process state, stated plainly because they matter operationally:

  * The service runs `--workers 2`, and each worker holds its OWN counter, so the
    effective ceiling is roughly 2x the configured limit. The limit is a blast
    shield, not a billing meter, so approximate is fine — but do not document the
    configured number to the client as an exact quota.
  * Counters reset when the service restarts. Also fine for a blast shield.

WHAT IS DELIBERATELY *NOT* LIMITED
----------------------------------
Stones, only requests. The desk legitimately prices a whole book in one call —
5,000 stones in a single request that takes minutes (`/frontoffice/price` caps
the batch itself). Throttling by stone count would punish the intended usage
while doing nothing about a retry loop, which is many SMALL requests.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

# Requests per window, per caller. Generous on purpose: a desk pricing books all
# day comes nowhere near this, so the limit only ever engages on a runaway loop.
DEFAULT_LIMIT = 120
DEFAULT_WINDOW_S = 60.0

# Cap on distinct callers tracked, so a spray of forged keys cannot grow this
# dict without bound. Evicting the coldest caller is safe: the worst case is that
# somebody who has been idle gets a fresh allowance.
_MAX_TRACKED = 2048

_hits: dict[str, deque[float]] = {}
_lock = threading.Lock()


def _limit() -> int:
    """Read from the environment each call so it is tunable without a redeploy."""
    try:
        return max(1, int(os.environ.get("GS_RATE_LIMIT", DEFAULT_LIMIT)))
    except (TypeError, ValueError):
        return DEFAULT_LIMIT


def _window() -> float:
    try:
        return max(1.0, float(os.environ.get("GS_RATE_WINDOW_S", DEFAULT_WINDOW_S)))
    except (TypeError, ValueError):
        return DEFAULT_WINDOW_S


def enabled() -> bool:
    """Off only if explicitly disabled — safe by default.

    `GS_RATE_LIMIT=0` turns it off, for a load test or a one-off bulk migration.
    """
    return _limit() > 0 and os.environ.get("GS_RATE_LIMIT", "").strip() != "0"


def check(caller: str, *, now: float | None = None) -> tuple[bool, int, float]:
    """Record a hit for `caller`. Returns (allowed, remaining, retry_after_s).

    `now` is injectable so the tests do not have to sleep through a real window.
    """
    if not enabled():
        return True, _limit(), 0.0
    t = time.monotonic() if now is None else now
    limit, window = _limit(), _window()

    with _lock:
        q = _hits.get(caller)
        if q is None:
            if len(_hits) >= _MAX_TRACKED:
                # Evict the caller whose most recent hit is oldest.
                coldest = min(_hits, key=lambda k: _hits[k][-1] if _hits[k] else 0.0)
                _hits.pop(coldest, None)
            q = _hits[caller] = deque()

        cutoff = t - window
        while q and q[0] <= cutoff:
            q.popleft()

        if len(q) >= limit:
            # Oldest hit in the window decides when a slot frees up.
            return False, 0, max(0.0, q[0] + window - t)

        q.append(t)
        return True, limit - len(q), 0.0


def reset() -> None:
    """Clear all counters. For tests, and for an operator who has just raised the
    limit and does not want to wait out the current window."""
    with _lock:
        _hits.clear()
