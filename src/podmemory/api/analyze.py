import logging
import time

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File

from ..core.config import settings
from ..schemas.analysis import AnalyzeRequest, AnalysisResponse
from ..services.analyzer import analyze_transcript, AnalysisError
from ..services.transcript import (
    is_youtube_url,
    get_youtube_transcript,
    from_user_paste,
)
from ..services.audio_transcript import transcribe_with_groq

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

_models_cache: list[dict] = []
_models_cache_ts: float = 0
CACHE_TTL = 3600

# Rate limiter for YouTube transcript API (datacenter IPs get banned easily)
_yt_last_request: float = 0
_YT_MIN_INTERVAL = 5  # seconds between YouTube requests


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


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio/video via Groq Whisper. Returns {text, source}."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 25MB)")
    try:
        result = await transcribe_with_groq(data, file.filename or "audio.mp3")
        return {"text": result["text"], "source": result["source"]}
    except RuntimeError as e:
        msg = str(e)
        if msg == "NO_GROQ_KEY":
            raise HTTPException(503, "GROQ_UNAVAILABLE")
        if msg == "GROQ_RATE_LIMIT":
            raise HTTPException(503, "GROQ_RATE_LIMIT")
        raise HTTPException(502, msg)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(req: AnalyzeRequest):
    if not req.url and not req.text:
        raise HTTPException(400, "Provide a video URL or transcript text")

    transcript = None

    if req.url and is_youtube_url(req.url):
        # Rate limit YouTube requests (datacenter IPs get banned easily)
        global _yt_last_request
        now = time.time()
        wait = _YT_MIN_INTERVAL - (now - _yt_last_request)
        if wait > 0:
            import asyncio
            await asyncio.sleep(wait)
        _yt_last_request = time.time()

        try:
            transcript = await get_youtube_transcript(req.url)
        except (ValueError, Exception) as e:
            err_msg = str(e)
            logger.warning("YouTube transcript failed for %s: %s", req.url, err_msg)
            raise HTTPException(
                400,
                "NO_SUBTITLES: This video has no subtitles available. "
                "Upload the audio/video file instead — it will be transcribed automatically."
            )

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

    if not transcript:
        raise HTTPException(502, "Failed to obtain transcript")

    try:
        result = await analyze_transcript(transcript, req.model)
    except AnalysisError as e:
        raise HTTPException(502, str(e))

    return AnalysisResponse(**result)
