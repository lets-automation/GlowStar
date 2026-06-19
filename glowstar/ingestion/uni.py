"""Uni Diamonds market connector (brief Section 3.3).

Builds the form-data filter for the export-report endpoint using the CONFIRMED
codebook (mappings.to_code, strict by default) so an unconfirmed code can never
silently query the wrong stones. Auth via env headers.
"""

from __future__ import annotations

from .http import request, require_env
from ..market.mappings import to_code

EXPORT_REPORT_URL = "https://app.uni.diamonds/api/1.03/free-form-price-analysis/export-report/1"


def _headers() -> dict:
    return {
        "deviceid": require_env("UNI_DEVICE_ID"),
        "token": require_env("UNI_TOKEN"),
        "platform": require_env("UNI_PLATFORM"),
    }


def build_filter(
    *, shape: str, size_from: float, size_to: float,
    colors: list[str], clarities: list[str],
    lab: str | None = None, country: str | None = None,
    fluorescence_code: int | None = None, search_type: int = 1,
    strict: bool = True,
) -> dict:
    """Build the export-report form-data body from human values.

    Numeric codes are resolved via the confirmed Uni codebook; pass strict=False
    only for experimentation with inferred (unverified) codes.
    """
    body: dict[str, str] = {
        "shape": str(to_code("shape", shape, strict=strict)),
        "size_from": f"{size_from:.2f}",
        "size_to": f"{size_to:.2f}",
        "search_type": str(search_type),
    }
    for i, c in enumerate(colors):
        body[f"color[{i}]"] = str(to_code("color", c, strict=strict))
    for i, cl in enumerate(clarities):
        body[f"clarity[{i}]"] = str(to_code("clarity", cl, strict=strict))
    if lab is not None:
        body["lab_ids[0]"] = str(to_code("lab", lab, strict=strict))
    if country is not None:
        body["country_id[0]"] = str(to_code("country", country, strict=strict))
    if fluorescence_code is not None:
        body["fluorescence_intensity[0]"] = str(fluorescence_code)
    return body


def fetch_market(filter_body: dict) -> list[dict]:
    """POST the export-report request; return the `data` list of market stones."""
    resp = request("POST", EXPORT_REPORT_URL, headers=_headers(), data=filter_body)
    return resp.json().get("data", [])
