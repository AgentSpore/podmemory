"""Audio transcription via Groq Whisper API + yt-dlp for audio download.

Primary: Groq Whisper (fast, high quality, free 10h/month)
Fallback: Browser Whisper.js (handled by frontend, no server dependency)

Note: yt-dlp download_youtube_audio is available but NOT auto-triggered
from /api/analyze — datacenter IPs get banned by YouTube.
Used only via explicit /api/youtube-audio/{video_id} endpoint.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import httpx
from loguru import logger

from ..core.config import settings


_WHISPER_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"]


async def transcribe_with_groq(audio_data: bytes, filename: str = "audio.mp3") -> dict:
    """Transcribe via Groq Whisper API. Tries turbo first, falls back to v3."""
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

            if resp.status_code in (429, 503):
                logger.warning("Groq {} returned {} — trying next model", model, resp.status_code)
                await asyncio.sleep(3)
                continue

            if resp.status_code != 200:
                err = resp.json().get("error", {}).get("message", resp.text[:200])
                raise RuntimeError(f"Groq error: {err}")

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
    """Download audio from any yt-dlp supported URL (YouTube, VK, Rutube, etc.).

    Uses ffmpeg postprocessor to extract audio-only (~2-5 MB instead of 100+ MB video).
    """
    def _download():
        import yt_dlp

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
            # Limit to first 15 min of audio
            "download_ranges": yt_dlp.utils.download_range_func(None, [(0, 900)]),
            "force_keyframes_at_cuts": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            for f in Path(tmp_dir).iterdir():
                if f.is_file() and f.suffix in (".mp3", ".m4a", ".ogg", ".opus", ".wav"):
                    return f
            filename = ydl.prepare_filename(info)
            result = Path(filename)
            if result.exists():
                return result
            mp3 = result.with_suffix(".mp3")
            if mp3.exists():
                return mp3
            raise RuntimeError("Download completed but audio file not found")

    return await asyncio.to_thread(_download)


async def download_youtube_audio(video_id: str) -> Path:
    """Download audio from YouTube (convenience wrapper)."""
    return await download_audio(f"https://www.youtube.com/watch?v={video_id}")


async def transcribe_url(url: str, platform: str = "unknown") -> dict:
    """Full server-side pipeline: yt-dlp download → Groq Whisper.

    Returns {text, source, language} or raises RuntimeError.
    """
    logger.info("Downloading audio from {}: {}", platform, url)
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
