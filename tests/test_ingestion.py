"""Tests for the ingestion layer: credentials, HTTPS/safe-logging, the four
connectors (with mocked HTTP), and immutable snapshots with schema drift.

No live network calls — connectors are exercised against fake responses so the
request-building and auth logic is verified deterministically.
"""

from __future__ import annotations

import json

import pytest

from glowstar.ingestion import http, diamanto, channel_partner, uni
from glowstar.ingestion.snapshots import save_snapshot
from glowstar.market.mappings import UnmappedCodeError


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


# --- credentials & safe logging ---

def test_require_env(monkeypatch):
    monkeypatch.delenv("FOO_X", raising=False)
    with pytest.raises(http.CredentialError):
        http.require_env("FOO_X")
    monkeypatch.setenv("FOO_X", "v")
    assert http.require_env("FOO_X") == "v"


def test_safe_url_strips_secrets():
    s = http.safe_url("https://api.example.com/GetAllRecord/user/pa$$word?x=1")
    assert "pa$$word" not in s and "user" not in s and "example.com" in s


def test_enforce_https_blocks_http():
    with pytest.raises(ValueError):
        http.enforce_https("http://insecure.example.com")


# --- connectors (mocked) ---

def test_diamanto_token(monkeypatch):
    for k in ("DIAMANTO_USERNAME", "DIAMANTO_PASSWORD", "DIAMANTO_CLIENT_ID", "DIAMANTO_UDID"):
        monkeypatch.setenv(k, "x")
    captured = {}

    def fake_request(method, url, **kw):
        captured.update(method=method, url=url, data=kw.get("data"))
        return FakeResp({"access_token": "TOK123"})

    monkeypatch.setattr(http, "requests", type("R", (), {"request": staticmethod(fake_request)}))
    monkeypatch.setattr(diamanto, "request", http.request)
    tok = diamanto.get_access_token()
    assert tok == "TOK123"
    assert captured["data"]["grant_type"] == "password"


def test_channel_partner_records(monkeypatch):
    monkeypatch.setenv("CHANNEL_PARTNER_USER", "u")
    monkeypatch.setenv("CHANNEL_PARTNER_PASS", "p")

    def fake_request(method, url, **kw):
        assert url.startswith("https://")
        return FakeResp([{"StoneId": "A"}, {"StoneId": "B"}])

    monkeypatch.setattr(http, "requests", type("R", (), {"request": staticmethod(fake_request)}))
    monkeypatch.setattr(channel_partner, "request", http.request)
    recs = channel_partner.get_all_records()
    assert len(recs) == 2


def test_uni_build_filter_uses_confirmed_codes():
    # Codes verified against the live API (clarity IF=2, VVS1=3, not the doc's 1/2).
    body = uni.build_filter(shape="Round", size_from=0.30, size_to=0.31,
                            colors=["D"], clarities=["IF", "VVS1"], lab="GIA", country="India",
                            fluorescence_code=7)
    assert body["shape"] == "1" and body["color[0]"] == "1"
    assert body["clarity[0]"] == "2" and body["clarity[1]"] == "3"
    assert body["lab_ids[0]"] == "1" and body["country_id[0]"] == "99"


def test_uni_build_filter_rejects_unmapped_strict():
    with pytest.raises(UnmappedCodeError):
        uni.build_filter(shape="Kite", size_from=1.0, size_to=1.5,
                         colors=["G"], clarities=["VS2"])   # rare shape, no verified code


# --- snapshots ---

def test_snapshot_is_immutable_and_idempotent(tmp_path, monkeypatch):
    import glowstar.ingestion.snapshots as snap
    monkeypatch.setattr(snap, "SNAPSHOT_ROOT", tmp_path)
    recs = [{"StoneId": "A", "Status": "Stock"}]
    r1 = save_snapshot(recs, "channel_partner", snapshot_date="2026-06-18")
    assert not r1.already_existed and r1.n_records == 1
    # Second save same day with different content must NOT overwrite.
    r2 = save_snapshot([{"StoneId": "A", "X": 1}], "channel_partner", snapshot_date="2026-06-18")
    assert r2.already_existed
    on_disk = json.loads((tmp_path / "channel_partner" / "2026-06-18.json").read_text())
    assert on_disk == recs


def test_snapshot_detects_schema_drift(tmp_path, monkeypatch):
    import glowstar.ingestion.snapshots as snap
    monkeypatch.setattr(snap, "SNAPSHOT_ROOT", tmp_path)
    save_snapshot([{"StoneId": "A", "Color": "G"}], "src", snapshot_date="2026-06-01")
    r = save_snapshot([{"StoneId": "A", "Color": "G", "BGM": "No"}], "src", snapshot_date="2026-06-02")
    assert "BGM" in r.added_fields
