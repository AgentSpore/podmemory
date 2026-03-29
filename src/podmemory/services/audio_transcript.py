"""Audio transcription: subtitles → Groq Whisper → browser Whisper fallback.

Pipeline for non-YouTube platforms:
1. fetch_subtitles() — yt-dlp metadata only, no download (~1-2s)
2. transcribe_with_groq() — yt-dlp audio + Groq Whisper API (~30-90s)
3. Browser Whisper.js — frontend fallback when Groq unavailable
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

import httpx
import yt_dlp
from loguru import logger

from ..core.config import settings

_WHISPER_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"]

# SRT/VTT noise: timestamps, sequence numbers, headers, HTML tags
_SUB_TIMESTAMP_RE = re.compile(r"\d{2}:\d{2}[:\.,]\d{2,3}\s*-->.*")
_SUB_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SUB_HEADERS = {"WEBVTT", "NOTE", "STYLE", "Kind:", "Language:"}


def _parse_subtitle_text(raw: str) -> str:
    """Parse SRT/VTT content into plain text."""
    lines: list[str] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.isdigit():
            continue
        if _SUB_TIMESTAMP_RE.match(line):
            continue
        if any(line.startswith(h) for h in _SUB_HEADERS):
            continue
        line = _SUB_HTML_TAG_RE.sub("", line)
        if line:
            lines.append(line)
    return " ".join(lines)


async def fetch_subtitles(url: str) -> dict | None:
    """Extract subtitles via yt-dlp metadata (no audio download).

    Returns {text, source, language} or None if unavailable.
    """
    def _fetch() -> dict | None:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "writesubtitles": True}) as ydl:
            info = ydl.extract_info(url, download=False)

        subs = info.get("subtitles") or {}
        if not subs:
            return None

        for lang, formats in subs.items():
            sub_url = next((f["url"] for f in formats if f["ext"] in ("srt", "vtt")), None)
            if not sub_url:
                continue

            resp = httpx.get(sub_url, timeout=10)
            if resp.status_code != 200 or len(resp.text) < 50:
                continue

            text = _parse_subtitle_text(resp.text)
            if len(text) < 20:
                continue

            return {"text": text, "source": f"subtitles_{lang}", "language": lang}

        return None

    return await asyncio.to_thread(_fetch)


async def transcribe_with_groq(audio_data: bytes, filename: str = "audio.mp3") -> dict:
    """Transcribe via Groq Whisper API. Cascades: turbo → v3 on 403/429/503."""
    fd, tmp_path = tempfile.mkstemp(suffix=Path(filename).suffix or ".mp3")
    tmp = Path(tmp_path)
    try:
        os.write(fd, audio_data)
        os.close(fd)

        for model in _WHISPER_MODELS:
            async with httpx.AsyncClient(timeout=120) as client:
                with open(tmp, "rb") as f:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                        files={"file": (tmp.name, f, "audio/mpeg")},
                        data={"model": model, "response_format": "verbose_json"},
                    )

            if resp.status_code in (403, 429, 503):
                logger.warning("Groq {} returned {} — trying next model", model, resp.status_code)
                await asyncio.sleep(3)
                continue

            if resp.status_code != 200:
                err = resp.json().get("error", {}).get("message", resp.text[:200])
                raise RuntimeError(f"Groq error ({resp.status_code}): {err}")

            data = resp.json()
            text = data.get("text", "")
            if not text:
                raise RuntimeError("Empty transcription")

            logger.info("Transcribed with {} ({} chars)", model, len(text))
            return {"text": text, "source": "groq_whisper", "language": data.get("language", "unknown")}

        raise RuntimeError("GROQ_RATE_LIMIT")
    finally:
        tmp.unlink(missing_ok=True)


async def download_audio(url: str) -> Path:
    """Download audio via yt-dlp + ffmpeg extraction (mp3 64kbps)."""
    def _download() -> Path:
        tmp_dir = tempfile.mkdtemp(prefix="podmemory_")
        output_path = str(Path(tmp_dir) / "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/worst",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "socket_timeout": 30,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
            "download_ranges": yt_dlp.utils.download_range_func(None, [(0, 900)]),
            "force_keyframes_at_cuts": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            for f in Path(tmp_dir).iterdir():
                if f.is_file() and f.suffix in (".mp3", ".m4a", ".ogg", ".opus", ".wav"):
                    return f
            prepared = Path(ydl.prepare_filename(info))
            for candidate in (prepared, prepared.with_suffix(".mp3")):
                if candidate.exists():
                    return candidate
            raise RuntimeError("Download completed but audio file not found")

    return await asyncio.to_thread(_download)


async def download_youtube_audio(video_id: str) -> Path:
    """Download audio from YouTube (convenience wrapper)."""
    return await download_audio(f"https://www.youtube.com/watch?v={video_id}")


async def transcribe_url(url: str, platform: str = "unknown") -> dict:
    """Subtitles (fast) → yt-dlp + Groq Whisper (fallback).

    Returns {text, source, language} or raises RuntimeError.
    """
    try:
        subs = await fetch_subtitles(url)
        if subs:
            logger.info("Got subtitles for {} ({} chars)", platform, len(subs["text"]))
            return subs
    except Exception as e:
        logger.debug("Subtitle fetch failed for {}: {}", url, e)

    logger.info("No subtitles, downloading audio from {}: {}", platform, url)
    audio_path = await download_audio(url)

    try:
        audio_data = audio_path.read_bytes()
        size_mb = len(audio_data) / (1024 * 1024)
        logger.info("Downloaded {:.1f} MB, transcribing with Groq...", size_mb)

        if size_mb > 25:
            raise RuntimeError("Audio too large (>25MB). Try a shorter video.")

        result = await transcribe_with_groq(audio_data, f"{platform}_audio.m4a")
        result["source"] = f"{platform}_groq_whisper"
        return result
    finally:
        try:
            audio_path.unlink(missing_ok=True)
            audio_path.parent.rmdir()
        except OSError:
            pass
