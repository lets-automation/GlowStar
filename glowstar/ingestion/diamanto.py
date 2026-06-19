"""Diamanto connectors: auth token (3.1) and internal grid history (3.2)."""

from __future__ import annotations

from .http import request, require_env

TOKEN_URL = "https://pricingapi.diamanto.co/api/token"
CELLS_HISTORY_URL = "https://pricingapi.diamanto.co/api/SpreadSheetCells/GetCellsHistory"


def get_access_token() -> str:
    """POST /token with env credentials; return the bearer access_token."""
    body = {
        "username": require_env("DIAMANTO_USERNAME"),
        "password": require_env("DIAMANTO_PASSWORD"),
        "grant_type": "password",
        "client_id": require_env("DIAMANTO_CLIENT_ID"),
        "userid": "",
        "udid": require_env("DIAMANTO_UDID"),
    }
    resp = request("POST", TOKEN_URL, data=body,
                   headers={"Content-Type": "application/x-www-form-urlencoded"})
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("Diamanto token response missing access_token")
    return token


def get_cells_history(from_date: str, to_date: str, token: str | None = None) -> list[dict]:
    """POST /GetCellsHistory for the client's internal grid history in a window.

    Dates: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'. Returns the cell-history list.
    """
    token = token or get_access_token()
    resp = request("POST", CELLS_HISTORY_URL,
                   headers={"Authorization": f"Bearer {token}"},
                   json={"fromDate": from_date, "toDate": to_date})
    return resp.json()
