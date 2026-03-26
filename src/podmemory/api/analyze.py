import time

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core.config import settings
from ..schemas.analysis import AnalyzeRequest, AnalysisResponse
from ..services.analyzer import analyze_transcript, AnalysisError
from ..services.transcript import (
    is_youtube_url,
    extract_youtube_id,
    get_youtube_transcript,
    from_user_paste,
)
from ..services.audio_transcript import download_youtube_audio

router = APIRouter(prefix="/api", tags=["analysis"])

_models_cache: list[dict] = []
_models_cache_ts: float = 0
CACHE_TTL = 3600


async def _fetch_free_models() -> list[dict]:
    global _models_cache, _models_cache_ts
    if _models_cache and time.time() - _models_cache_ts < CACHE_TTL:
        return _models_cache
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://openrouter.ai/api/v1/models")
            data = resp.json()
            models = []
            for m in data.get("data", []):
                pricing = m.get("pricing", {})
                if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                    modalities = m.get("architecture", {}).get("output_modalities", [])
                    if "text" not in modalities:
                        continue
                    models.append({
                        "id": m["id"],
                        "name": m.get("name", m["id"]),
                        "context_length": m.get("context_length", 0),
                    })
            models.sort(key=lambda x: x["name"])
            _models_cache = models
            _models_cache_ts = time.time()
            return models
    except Exception:
        return _models_cache or []


@router.get("/config")
async def get_config():
    models = await _fetch_free_models()
    default = settings.llm_model
    free_ids = {m["id"] for m in models}
    if default not in free_ids and models:
        default = models[0]["id"]
    return {
        "default_model": default,
        "models": models,
        "has_api_key": bool(settings.openrouter_api_key),
    }


@router.get("/youtube-audio/{video_id}")
async def get_youtube_audio(video_id: str):
    """Download audio from YouTube for client-side Whisper.js transcription.
    Used as fallback when subtitles are unavailable.
    """
    try:
        audio_path = await download_youtube_audio(video_id)
        return FileResponse(
            audio_path,
            media_type="audio/mp4",
            filename=f"{video_id}.m4a",
        )
    except Exception as e:
        raise HTTPException(502, f"Failed to download audio: {e}")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(req: AnalyzeRequest):
    if not req.url and not req.text:
        raise HTTPException(400, "Provide a video URL or transcript text")

    try:
        if req.url and is_youtube_url(req.url):
            transcript = await get_youtube_transcript(req.url)
        elif req.text:
            transcript = from_user_paste(req.text)
        elif req.url:
            raise HTTPException(
                400,
                "Only YouTube URLs are supported for auto-transcription. "
                "For other videos, upload the audio/video file."
            )
        else:
            raise HTTPException(400, "No input provided")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Failed to process: {e}")

    try:
        result = await analyze_transcript(transcript, req.model)
    except AnalysisError as e:
        raise HTTPException(502, str(e))

    return AnalysisResponse(**result)
