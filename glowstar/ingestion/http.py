"""Shared HTTP helpers: env credentials, HTTPS enforcement, retry, safe logging.

No secret or full URL is ever logged. Credentials are read lazily from the
environment so importing this module never requires them to be set.
"""

from __future__ import annotations

import logging
import os
import time
from urllib.parse import urlsplit

import requests

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 45      # full-bracket market pulls can be large; give them room
MAX_RETRIES = 3           # before falling back to banked data
BACKOFF_BASE = 1.5


class CredentialError(RuntimeError):
    """A required credential is missing from the environment."""


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise CredentialError(
            f"Missing required environment variable {name!r}. "
            "Set it in .env (see .env.example); never hardcode secrets."
        )
    return val


def safe_url(url: str) -> str:
    """Host + path only — strips query, credentials, and path-embedded secrets."""
    p = urlsplit(url)
    return f"{p.scheme}://{p.hostname}/…"


def enforce_https(url: str) -> None:
    if urlsplit(url).scheme != "https":
        raise ValueError(f"Refusing non-HTTPS request to {safe_url(url)}")


def request(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP with HTTPS enforcement, timeout, and exponential-backoff retry.

    Retries on connection errors and 5xx; raises for client errors. Logs only
    the safe URL, never the body or full URL.
    """
    enforce_https(url)
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    last_info = "unknown error"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} server error")
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            # SANITIZE: a raw requests exception can embed the FULL url — including
            # path-embedded credentials (Channel Partner puts user/pass in the
            # path). Keep only the safe status/type; never let the exception (or its
            # chain) propagate into a log.exception traceback.
            status = getattr(getattr(e, "response", None), "status_code", None)
            last_info = type(e).__name__ + (f" ({status})" if status else "")
            if attempt == MAX_RETRIES or _is_client_error(e):
                break
            sleep = BACKOFF_BASE ** attempt
            log.warning("Request to %s failed (%s); retry %d/%d in %.1fs",
                        safe_url(url), type(e).__name__, attempt, MAX_RETRIES, sleep)
            time.sleep(sleep)
    # `from None` drops the chained cause so the credential-bearing URL can never
    # reach a traceback; the message itself is already host+path-only via safe_url.
    raise RuntimeError(
        f"Request to {safe_url(url)} failed after {MAX_RETRIES} attempts ({last_info})") from None


def _is_client_error(e: Exception) -> bool:
    resp = getattr(e, "response", None)
    return resp is not None and 400 <= resp.status_code < 500
