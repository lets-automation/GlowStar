"""Empirically calibrate the Uni request codebook from the LIVE API.

The Uni filter uses numeric codes; the docs only proved a few. Rather than guess,
we probe the live API: set one field to a candidate code, and read back the value
the API actually returns. The verified mapping is written to
artifacts/uni_codebook.json and loaded by market.mappings — turning guesses into
facts confirmed against the live endpoint.

Run:  python -m glowstar.market.calibrate_codebook
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter

from .. import config  # noqa: F401  (loads .env)
from ..config import ARTIFACTS_DIR
from ..ingestion import uni

log = logging.getLogger(__name__)

# (field, response_key, candidate codes, base filter overrides)
# Every probe keeps a TIGHT color+clarity+size filter so responses stay small
# and fast — a broad query (e.g. all rounds) returns a huge payload and times out.
_TIGHT = {"color[0]": "4", "clarity[0]": "5", "size_from": "0.90", "size_to": "1.10"}
_PROBES = [
    ("shape", "shape", range(1, 16), {**_TIGHT}),
    ("color", "color", range(1, 12), {"shape": "1", "clarity[0]": "5", "size_from": "0.90", "size_to": "1.10"}),
    ("clarity", "clarity", range(1, 12), {"shape": "1", "color[0]": "4", "size_from": "0.90", "size_to": "1.10"}),
    ("lab", "lab", range(1, 6), {"shape": "1", **_TIGHT}),
    ("fluorescence", "fluorescence", range(1, 10), {"shape": "1", **_TIGHT}),
]

_FIELD_PARAM = {"shape": "shape", "color": "color[0]", "clarity": "clarity[0]",
                "lab": "lab_ids[0]", "fluorescence": "fluorescence_intensity[0]"}


def _dominant(stones: list[dict], key: str) -> tuple[str | None, int]:
    c = Counter(s.get(key) for s in stones if s.get(key) not in (None, ""))
    return (c.most_common(1)[0][0], len(stones)) if c else (None, len(stones))


def calibrate(delay: float = 0.2) -> dict:
    codebook: dict[str, dict[str, int]] = {}
    for field, resp_key, codes, base in _PROBES:
        mapping: dict[str, int] = {}
        for code in codes:
            body = {"search_type": "1", **base, _FIELD_PARAM[field]: str(code)}
            try:
                stones = uni.fetch_market(body)
            except Exception as e:
                log.warning("probe %s=%s failed: %s", field, code, str(e)[:80])
                continue
            value, n = _dominant(stones, resp_key)
            if value is not None and n > 0:
                # First code that returns a value wins it (codes are 1:1 with values).
                mapping.setdefault(str(value), code)
                log.info("%s code %s -> %r (n=%s)", field, code, value, n)
            time.sleep(delay)
        codebook[field] = mapping
    return codebook


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cb = calibrate()
    out = ARTIFACTS_DIR / "uni_codebook.json"
    out.write_text(json.dumps(cb, indent=2), encoding="utf-8")
    log.info("Wrote verified codebook -> %s", out)
    for field, m in cb.items():
        log.info("  %s: %s", field, m)


if __name__ == "__main__":
    main()
