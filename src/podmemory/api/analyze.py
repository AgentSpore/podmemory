import asyncio
import random
import re
import tempfile
import time
from pathlib import Path

import genanki
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from loguru import logger

from ..core.config import settings
from ..schemas.analysis import AnalyzeRequest, AnalysisResponse, AnkiExportRequest
from ..services.analyzer import analyze_transcript, AnalysisError
from ..services.transcript import (
    detect_platform,
    get_youtube_transcript,
    from_user_paste,
)
from ..services.audio_transcript import transcribe_with_groq, download_youtube_audio, transcribe_url

router = APIRouter(prefix="/api", tags=["analysis"])

_models_cache: list[dict] = []
_models_cache_ts: float = 0
CACHE_TTL = 3600

_yt_lock = asyncio.Lock()
_yt_last_request: float = 0
_YT_MIN_INTERVAL = 5

# YouTube video ID: exactly 11 chars, alphanumeric + dash + underscore
_YT_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")


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
    except Exception as e:
        logger.warning("Failed to fetch models: {}", e)
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
    """Download audio from YouTube via yt-dlp."""
    if not _YT_VIDEO_ID_RE.match(video_id):
        raise HTTPException(400, "Invalid video ID")
    try:
        audio_path = await download_youtube_audio(video_id)
        return FileResponse(audio_path, media_type="audio/mp4", filename=f"{video_id}.m4a")
    except Exception as e:
        logger.error("YouTube audio download failed for {}: {}", video_id, e)
        raise HTTPException(502, "Failed to download audio")


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio/video via Groq Whisper. Returns {text, source}."""
    if file.size and file.size > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 25MB)")
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
        logger.error("Transcription failed: {}", e)
        raise HTTPException(502, "Transcription failed")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(req: AnalyzeRequest):
    if not req.url and not req.text:
        raise HTTPException(400, "Provide a video URL or transcript text")

    transcript = None
    platform = detect_platform(req.url) if req.url else None

    if req.url and platform == "youtube":
        global _yt_last_request
        async with _yt_lock:
            now = time.time()
            wait = _YT_MIN_INTERVAL - (now - _yt_last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            _yt_last_request = time.time()

        try:
            transcript = await get_youtube_transcript(req.url)
        except (ValueError, Exception) as e:
            logger.warning("YouTube subtitles failed for {}: {}", req.url, e)
            raise HTTPException(
                400,
                "NO_SUBTITLES: This video has no subtitles available. "
                "Upload the audio/video file instead — it will be transcribed automatically."
            )

    elif req.url and platform:
        try:
            result = await transcribe_url(req.url, platform)
            transcript = from_user_paste(result["text"], result.get("language", "unknown"))
            transcript.source = result["source"]
        except RuntimeError as e:
            msg = str(e)
            if msg in ("NO_GROQ_KEY", "GROQ_RATE_LIMIT"):
                raise HTTPException(503, f"GROQ_UNAVAILABLE: {msg}")
            logger.error("Transcription failed for {} ({}): {}", req.url, platform, e)
            raise HTTPException(502, f"Failed to transcribe {platform} video")

    elif req.text:
        transcript = from_user_paste(req.text)
    elif req.url:
        raise HTTPException(
            400,
            "Unsupported URL. Supported: YouTube, VK, Rutube, TikTok, Twitch, Vimeo, and more. "
            "For other videos, upload the audio/video file."
        )
    else:
        raise HTTPException(400, "No input provided")

    if not transcript:
        raise HTTPException(502, "Failed to obtain transcript")

    try:
        result = await analyze_transcript(transcript, req.model)
    except AnalysisError as e:
        logger.error("Analysis failed: {}", e)
        raise HTTPException(502, "Analysis failed. Try a different model.")

    return AnalysisResponse(**result)


@router.post("/export/anki")
async def export_anki(req: AnkiExportRequest):
    """Export flashcards as Anki .apkg file."""
    if not req.flashcards:
        raise HTTPException(400, "No flashcards to export")

    model_id = random.randrange(1 << 30, 1 << 31)
    deck_id = random.randrange(1 << 30, 1 << 31)

    anki_model = genanki.Model(
        model_id,
        "PodMemory",
        fields=[{"name": "Question"}, {"name": "Answer"}],
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Question}}",
            "afmt": '{{FrontSide}}<hr id="answer">{{Answer}}',
        }],
    )

    deck = genanki.Deck(deck_id, req.title)
    for fc in req.flashcards:
        note = genanki.Note(model=anki_model, fields=[fc.q, fc.a])
        deck.add_note(note)

    tmp = tempfile.NamedTemporaryFile(suffix=".apkg", delete=False)
    try:
        genanki.Package(deck).write_to_file(tmp.name)
        tmp.close()
        data = Path(tmp.name).read_bytes()
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    safe_title = "".join(c for c in req.title if c.isalnum() or c in " -_")[:50].strip() or "podmemory"
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.apkg"'},
    )
