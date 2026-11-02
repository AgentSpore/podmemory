import asyncio
import json as json_lib
import random
import re
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import AsyncGenerator

import genanki
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, Response
from loguru import logger
from sse_starlette.sse import EventSourceResponse
from starlette.types import Receive, Scope, Send

from ..core.config import settings
from ..schemas.analysis import AnalyzeRequest, AnalysisResponse, AnkiExportRequest
from ..services.analyzer import analyze_transcript, AnalysisError
from ..services.model_catalog import fetch_free_models
from ..services.transcript import (
    detect_platform,
    get_youtube_transcript,
    from_user_paste,
)
from ..services.audio_transcript import download_audio, fetch_subtitles, transcribe_with_groq, download_youtube_audio, transcribe_url
from ..services.text_extract import extract_from_url, extract_from_pdf, extract_from_epub

router = APIRouter(prefix="/api", tags=["analysis"])

_yt_lock = asyncio.Lock()
_yt_last_request: float = 0
_YT_MIN_INTERVAL = 5

# YouTube video ID: exactly 11 chars, alphanumeric + dash + underscore
_YT_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# Temp audio files for browser Whisper fallback (when Groq is unavailable)
import uuid
_TEMP_AUDIO_TTL = 300  # Browser fetches immediately; abandoned offers expire in five minutes.
_TEMP_AUDIO_MAX_FILES = 8
_TEMP_AUDIO_MAX_BYTES = 50 * 1024 * 1024
_MAX_TRANSCRIPTION_BYTES = 25 * 1024 * 1024


@dataclass
class _AudioOwner:
    path: Path
    size: int
    retry: asyncio.TimerHandle | None = None


# Pending links own an expiry timer; a GET consumes only this one-shot eligibility.
_temp_audio: dict[str, tuple[Path, asyncio.TimerHandle]] = {}
# Count and bytes include pending AND claimed/streaming files until removal succeeds.
_audio_owners: dict[str, _AudioOwner] = {}


def _remove_audio_file(path: Path) -> bool:
    """Return whether the audio was removed; an empty-directory failure is harmless."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    try:
        path.parent.rmdir()
    except OSError:
        pass
    return True


def _release_temp_audio(audio_id: str) -> None:
    pending = _temp_audio.pop(audio_id, None)
    if pending is not None:
        pending[1].cancel()
    owner = _audio_owners.get(audio_id)
    if owner is None:
        return
    if owner.retry is not None:
        owner.retry.cancel()
    if _remove_audio_file(owner.path):
        _audio_owners.pop(audio_id)
    else:
        # Keep failed deletions charged to the budget; never expose paths/tokens.
        logger.warning("Temporary audio cleanup failed; capacity remains reserved")
        owner.retry = asyncio.get_running_loop().call_later(1, _release_temp_audio, audio_id)


def _expire_temp_audio(audio_id: str) -> None:
    if audio_id in _temp_audio:
        _release_temp_audio(audio_id)


def _offer_temp_audio(path: Path) -> str | None:
    size = path.stat().st_size
    if (len(_audio_owners) >= _TEMP_AUDIO_MAX_FILES
            or sum(owner.size for owner in _audio_owners.values()) + size > _TEMP_AUDIO_MAX_BYTES):
        return None
    # No await between admission and registration: concurrent requests cannot overbook.
    audio_id = uuid.uuid4().hex[:12]
    timer = asyncio.get_running_loop().call_later(
        _TEMP_AUDIO_TTL, _expire_temp_audio, audio_id
    )
    _audio_owners[audio_id] = _AudioOwner(path, size)
    _temp_audio[audio_id] = (path, timer)
    return audio_id


class _TemporaryAudioResponse(FileResponse):
    """Keep capacity reserved through sending, including cancellation/range errors."""

    def __init__(self, path: Path, audio_id: str):
        super().__init__(path, media_type="audio/mpeg", filename="audio.mp3")
        self.audio_id = audio_id

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            _release_temp_audio(self.audio_id)


@router.get("/config")
async def get_config():
    models = await fetch_free_models()
    default = settings.llm_model
    free_ids = {m["id"] for m in models}
    if default not in free_ids and models:
        default = models[0]["id"]
    return {
        "default_model": default,
        "models": models,
        "has_api_key": bool(settings.openrouter_api_key),
    }


@router.get("/temp-audio/{audio_id}")
async def get_temp_audio(audio_id: str):
    """Serve temp audio file for browser Whisper fallback."""
    pending = _temp_audio.pop(audio_id, None)
    if pending is None:
        raise HTTPException(404, "Audio not found or already downloaded")
    path, timer = pending
    timer.cancel()
    if not path.is_file():
        _release_temp_audio(audio_id)
        raise HTTPException(404, "Audio not found or already downloaded")
    return _TemporaryAudioResponse(path, audio_id)


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


@router.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extract text from PDF. Returns {text, title, word_count, chapters}."""
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50MB)")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50MB)")
    try:
        extracted = await extract_from_pdf(data, file.filename or "document.pdf")
        return {
            "text": extracted.text,
            "title": extracted.title,
            "word_count": extracted.word_count,
            "chapters": len(extracted.chapters),
            "source_type": "pdf",
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("PDF extraction failed: {}", e)
        raise HTTPException(502, "Failed to extract PDF text")


@router.post("/extract-epub")
async def extract_epub_file(file: UploadFile = File(...)):
    """Extract text from EPUB with chapter structure.

    Returns {text, title, author, word_count, chapters: [{title, text, word_count}]}.
    """
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 100MB)")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 100MB)")
    try:
        extracted = await extract_from_epub(data, file.filename or "book.epub")
        return {
            "text": extracted.text,
            "title": extracted.title,
            "author": extracted.author,
            "word_count": extracted.word_count,
            "source_type": "book",
            "chapters": [
                {"title": ch["title"], "text": ch["text"], "word_count": len(ch["text"].split())}
                for ch in extracted.chapters
            ],
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("EPUB extraction failed: {}", e)
        raise HTTPException(502, "Failed to extract EPUB text")


@router.post("/analyze-chapter", response_model=AnalysisResponse)
async def analyze_chapter(req: AnalyzeRequest):
    """Analyze a single book chapter. Send text + source_type='book'."""
    if not req.text:
        raise HTTPException(400, "Provide chapter text")
    transcript = from_user_paste(req.text)
    transcript.source = "book_chapter"
    try:
        result = await analyze_transcript(transcript, req.model, source_type=req.source_type or "book")
    except AnalysisError as e:
        logger.error("Chapter analysis failed: {}", e)
        raise HTTPException(502, "Analysis failed. Try a different model.")
    return AnalysisResponse(**result)


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
    source_type = req.source_type or ""
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

    elif req.url and not platform:
        # Not a video platform — try as article URL
        try:
            extracted = await extract_from_url(req.url)
            transcript = from_user_paste(extracted.text, extracted.language)
            transcript.source = f"article:{req.url}"
            source_type = "article"
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.error("Article extraction failed for {}: {}", req.url, e)
            raise HTTPException(502, "Failed to extract article text")

    elif req.text:
        transcript = from_user_paste(req.text)
    else:
        raise HTTPException(400, "No input provided")

    if not transcript:
        raise HTTPException(502, "Failed to obtain transcript")

    try:
        result = await analyze_transcript(transcript, req.model, source_type=source_type)
    except AnalysisError as e:
        logger.error("Analysis failed: {}", e)
        raise HTTPException(502, "Analysis failed. Try a different model.")

    return AnalysisResponse(**result)


def _sse_event(stage: str, progress: int, message: str, data: dict | None = None) -> str:
    payload = {"stage": stage, "progress": progress, "message": message}
    if data:
        payload["data"] = data
    return json_lib.dumps(payload, ensure_ascii=False)


@router.post("/analyze-stream")
async def analyze_video_stream(req: AnalyzeRequest):
    """SSE endpoint — streams progress stages during analysis."""

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            if not req.url and not req.text:
                yield _sse_event("error", 0, "Provide a video URL or transcript text")
                return

            transcript = None
            source_type = req.source_type or ""
            platform = detect_platform(req.url) if req.url else None

            if req.url and platform == "youtube":
                yield _sse_event("fetching_subtitles", 10, "Fetching YouTube subtitles...")

                global _yt_last_request
                async with _yt_lock:
                    now = time.time()
                    wait = _YT_MIN_INTERVAL - (now - _yt_last_request)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    _yt_last_request = time.time()

                try:
                    transcript = await get_youtube_transcript(req.url)
                    yield _sse_event("subtitles_ready", 30, "Subtitles loaded")
                except (ValueError, Exception):
                    yield _sse_event("error", 0, "NO_SUBTITLES: No subtitles available. Upload the audio/video file instead.")
                    return

            elif req.url and platform:
                # Try subtitles first (instant, ~1-2s)
                yield _sse_event("fetching_subtitles", 10, f"Checking {platform} subtitles...")
                try:
                    subs = await fetch_subtitles(req.url)
                    if subs:
                        yield _sse_event("subtitles_ready", 30, f"Subtitles loaded ({len(subs['text'])} chars)")
                        transcript = from_user_paste(subs["text"], subs.get("language", "unknown"))
                        transcript.source = subs["source"]
                except Exception as e:
                    logger.debug("Subtitle fetch failed: {}", e)

                if not transcript:
                    yield _sse_event("downloading_audio", 15, f"No subtitles. Downloading audio from {platform}...")
                    try:
                        audio_path = await download_audio(req.url)
                    except Exception as e:
                        logger.error("Download failed for {} ({}): {}", req.url, platform, e)
                        yield _sse_event("error", 0, f"Failed to download {platform} video")
                        return

                    audio_transferred = False
                    try:
                        yield _sse_event("transcribing", 40, "Transcribing audio with Whisper...")
                        audio_data = audio_path.read_bytes()
                        if len(audio_data) > _MAX_TRANSCRIPTION_BYTES:
                            yield _sse_event("error", 0, "Audio too large (>25MB). Try a shorter video.")
                            return
                        result = await transcribe_with_groq(audio_data, f"{platform}_audio.m4a")
                        result["source"] = f"{platform}_groq_whisper"
                        transcript = from_user_paste(result["text"], result.get("language", "unknown"))
                        transcript.source = result["source"]
                        yield _sse_event("transcription_ready", 60, f"Transcribed ({len(result['text'])} chars)")
                    except RuntimeError as e:
                        msg = str(e)
                        if "GROQ_RATE_LIMIT" in msg or "Forbidden" in msg or "403" in msg:
                            audio_id = _offer_temp_audio(audio_path)
                            if audio_id is None:
                                # Refusal must not retain bytes while a slow client holds this yield.
                                _remove_audio_file(audio_path)
                                yield _sse_event("error", 0, "Browser transcription is busy. Please try again shortly.")
                                return
                            audio_transferred = True
                            yield _sse_event("groq_fallback", 45, "Server transcription unavailable. Switching to browser...", {"audio_url": f"/api/temp-audio/{audio_id}"})
                            return
                        yield _sse_event("error", 0, f"Transcription failed: {msg}")
                        return
                    finally:
                        if not audio_transferred:
                            _remove_audio_file(audio_path)

            elif req.url and not platform:
                # Article URL — extract text via trafilatura
                yield _sse_event("extracting_article", 10, "Extracting article text...")
                source_type = "article"
                try:
                    extracted = await extract_from_url(req.url)
                    transcript = from_user_paste(extracted.text, extracted.language)
                    transcript.source = f"article:{req.url}"
                    yield _sse_event("article_ready", 30, f"Extracted ({extracted.word_count} words)")
                except ValueError as e:
                    yield _sse_event("error", 0, str(e))
                    return
                except Exception as e:
                    logger.error("Article extraction failed: {}", e)
                    yield _sse_event("error", 0, "Failed to extract article text")
                    return

            elif req.text:
                transcript = from_user_paste(req.text)
                source_type = req.source_type or "text"
                yield _sse_event("text_ready", 30, "Text received")

            else:
                yield _sse_event("error", 0, "No input provided")
                return

            if not transcript:
                yield _sse_event("error", 0, "Failed to obtain transcript")
                return

            yield _sse_event("analyzing", 70, "AI is analyzing the content...")

            try:
                result = await analyze_transcript(transcript, req.model, source_type=source_type)
            except AnalysisError as e:
                logger.error("Analysis failed: {}", e)
                yield _sse_event("error", 0, "Analysis failed. Try a different model.")
                return

            yield _sse_event("done", 100, "Analysis complete", result)

        except Exception as e:
            logger.error("Stream error: {}", e)
            yield _sse_event("error", 0, "Internal error")

    return EventSourceResponse(event_generator())


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

    # HTTP headers only support latin-1; use ASCII-safe filename + RFC 5987 for unicode
    ascii_title = "".join(c for c in req.title if c.isascii() and (c.isalnum() or c in " -_"))[:50].strip() or "podmemory"
    from urllib.parse import quote
    utf8_title = quote(req.title[:50], safe="")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_title}.apkg\"; filename*=UTF-8''{utf8_title}.apkg",
        },
    )
