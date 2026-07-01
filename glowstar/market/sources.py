"""Pluggable market-data sources (brief Section 3.3; client Q: "is Uni the only feed?").

Honest position encoded in code: today the live feed is Uni Diamonds. ALL polished
feeds (Uni, RapNet, IDEX, Nivoda) are *asking*-price feeds and all carry the same
"virtual inventory" duplication — the 6.2GB Uni dump was ~90% re-listings — so
adding more shallow feeds is not the accuracy lever; the client's OWN realized
sales are. This seam exists so the single highest-value add (RapNet — the client
already subscribes, and it is the source of the RAPI index) can be wired in as a
LOCALIZED change without touching the engine, if/when API access is provisioned.

Every source returns a `MarketTables` (segment medians + BGM deltas) in the exact
shape the engine consumes, so the engine stays source-agnostic. A source that is
not provisioned FAILS LOUD with how to enable it — it never returns fake/empty
market data.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from .anchor import MarketTables
from .live import LiveMarket


@runtime_checkable
class MarketSource(Protocol):
    name: str

    def build_tables(self, stones: pd.DataFrame,
                     base: MarketTables | None = None) -> MarketTables:
        ...


class UniMarketSource:
    """Live Uni Diamonds comparables — the working default."""

    name = "uni"

    def __init__(self, **kw):
        self._live = LiveMarket(**kw)

    def build_tables(self, stones: pd.DataFrame,
                     base: MarketTables | None = None) -> MarketTables:
        return self._live.build_tables(stones, base=base)

    @property
    def live(self) -> LiveMarket:
        return self._live


class _UnavailableSource:
    """A documented, not-yet-provisioned feed. Fails loud with how to enable it."""

    def __init__(self, name: str, how: str):
        self.name = name
        self._how = how

    def build_tables(self, stones: pd.DataFrame,
                     base: MarketTables | None = None) -> MarketTables:
        raise NotImplementedError(f"Market source {self.name!r} is not wired yet. {self._how}")


def RapNetMarketSource() -> _UnavailableSource:
    return _UnavailableSource(
        "rapnet",
        "RapNet is the deepest pool and the source of the RAPI index; the client "
        "already subscribes. Provision RapNet API access (Instant Inventory / Price "
        "List API), then implement build_tables() mirroring UniMarketSource.")


def IdexMarketSource() -> _UnavailableSource:
    return _UnavailableSource(
        "idex",
        "IDEX is an alternative asking-price feed (IDEX Diamond Index since 2004). "
        "Provision IDEX data access, then implement build_tables() here.")


_SOURCES = {"uni": UniMarketSource, "rapnet": RapNetMarketSource, "idex": IdexMarketSource}


def get_market_source(name: str = "uni", **kw) -> MarketSource:
    """Factory: return the named market source (default = the live Uni feed)."""
    key = (name or "uni").strip().lower()
    if key not in _SOURCES:
        raise ValueError(f"Unknown market source {name!r}; available: {sorted(_SOURCES)}")
    return _SOURCES[key](**kw) if key == "uni" else _SOURCES[key]()
