"""Channel Partner connector: inventory + sales (brief Section 3.4).

Credentials are passed in the URL path by this API, so we enforce HTTPS and use
safe_url() everywhere — the full URL with embedded credentials is never logged.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from .http import request, require_env, safe_url

BASE = "https://channelpartnerapi.azurewebsites.net/api/ChannelPartner/GetAllRecord"

log = logging.getLogger(__name__)


def get_all_records() -> list[dict]:
    """GET all inventory+sales records (single full-snapshot pull).

    Returns the full list (~28k records). Sales vs stock are distinguished by
    each record's `Status` downstream (loaders), not here.
    """
    user = quote(require_env("CHANNEL_PARTNER_USER"), safe="")
    pw = quote(require_env("CHANNEL_PARTNER_PASS"), safe="")
    url = f"{BASE}/{user}/{pw}"
    log.info("Pulling full record snapshot from %s", safe_url(url))
    resp = request("GET", url)
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("GetAllRecord did not return a list")
    log.info("Pulled %s records", f"{len(data):,}")
    return data
