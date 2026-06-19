"""Optional REST wrapper around PricingService (brief Section 6).

FastAPI gives an OpenAPI spec for free at /docs. The web framework is an
optional dependency — install with `pip install fastapi uvicorn` and run:
    uvicorn glowstar.service.app:app --reload
The pricing logic lives in PricingService and is fully usable without a server.
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - server is optional
    FastAPI = None

from .pricing_service import PricingService, StoneIn

if FastAPI is not None:
    app = FastAPI(title="Glow Star Pricing Engine", version="0.1.0")
    _service: PricingService | None = None

    def _get_service() -> PricingService:
        global _service
        if _service is None:
            _service = PricingService()
        return _service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/price")
    def price(stone: StoneIn) -> dict:
        """Price a single stone; returns suggestion + interval + guarded explanation."""
        return _get_service().price(stone)
else:  # pragma: no cover
    app = None
