"""Share the fresh free text-model catalog between configuration and analysis."""

import asyncio
import time

import httpx
from loguru import logger

_models_cache: list[dict] | None = None
_models_cache_ts: float = 0
CACHE_TTL = 3600
_RETRY_TTL = 5
_models_lock = asyncio.Lock()


async def fetch_free_models() -> list[dict]:
    """Return freshly verified free text models; never authorize stale entries."""
    global _models_cache, _models_cache_ts
    async with _models_lock:
        ttl = CACHE_TTL if _models_cache else _RETRY_TTL
        if _models_cache is not None and time.monotonic() - _models_cache_ts < ttl:
            return _models_cache
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get("https://openrouter.ai/api/v1/models")
                resp.raise_for_status()
                data = resp.json()
                models = []
                for m in data.get("data", []):
                    pricing = m.get("pricing", {})
                    if (
                        pricing.get("prompt") == "0"
                        and pricing.get("completion") == "0"
                    ):
                        modalities = m.get("architecture", {}).get(
                            "output_modalities", []
                        )
                        if "text" not in modalities:
                            continue
                        models.append(
                            {
                                "id": m["id"],
                                "name": m.get("name", m["id"]),
                                "context_length": m.get("context_length", 0),
                            }
                        )
                models.sort(key=lambda x: x["name"])
                _models_cache = models
                _models_cache_ts = time.monotonic()
                return models
        except (
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
        ) as exc:
            logger.warning("Failed to fetch models: {}", type(exc).__name__)
            _models_cache, _models_cache_ts = [], time.monotonic()
            return _models_cache


async def filter_allowed_models(models: list[str], default: str) -> list[str]:
    """Keep the operator default and models in the fresh public catalog."""
    free_ids = (
        {entry["id"] for entry in await fetch_free_models()}
        if any(model != default for model in models)
        else set()
    )
    return [model for model in models if model == default or model in free_ids]
