"""Audio transcription via Groq Whisper API + yt-dlp for audio download.

Primary: Groq Whisper (fast, high quality, free 10h/month)
Fallback: Browser Whisper.js (handled by frontend, no server dependency)

Note: yt-dlp download_youtube_audio is available but NOT auto-triggered
from /api/analyze — datacenter IPs get banned by YouTube.
Used only via explicit /api/youtube-audio/{video_id} endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


async def transcribe_with_groq(audio_data: bytes, filename: str = "audio.mp3") -> dict:
    """Transcribe via Groq Whisper API. Returns {text, source, language}."""
    if not settings.groq_api_key:
        raise RuntimeError("NO_GROQ_KEY")

    tmp = Path(tempfile.mktemp(suffix=Path(filename).suffix or ".mp3"))
    tmp.write_bytes(audio_data)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            with open(tmp, "rb") as f:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                    files={"file": (tmp.name, f, "audio/mpeg")},
                    data={"model": "whisper-large-v3", "response_format": "verbose_json"},
                )

        if resp.status_code == 429:
            raise RuntimeError("GROQ_RATE_LIMIT")

        if resp.status_code != 200:
            err = resp.json().get("error", {}).get("message", resp.text[:200])
            raise RuntimeError(f"Groq error: {err}")

        data = resp.json()
        text = data.get("text", "")
        if not text:
            raise RuntimeError("Empty transcription")

        return {"text": text, "source": "groq_whisper", "language": data.get("language", "unknown")}
    finally:
        tmp.unlink(missing_ok=True)


async def download_youtube_audio(video_id: str) -> Path:
    """Download audio from YouTube via yt-dlp (fast, reliable).

    Warning: may fail on datacenter IPs (YouTube blocks them).
    """
    def _download():
        import yt_dlp

        tmp_dir = tempfile.mkdtemp(prefix="podmemory_")
        output_path = str(Path(tmp_dir) / "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "socket_timeout": 30,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=True,
            )
            filename = ydl.prepare_filename(info)
            result = Path(filename)
            if not result.exists():
                for f in Path(tmp_dir).iterdir():
                    if f.is_file():
                        return f
                raise RuntimeError("Download completed but file not found")
            return result

    return await asyncio.to_thread(_download)
