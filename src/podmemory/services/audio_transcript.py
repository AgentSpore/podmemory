"""Audio transcription via Groq Whisper API.

Primary: Groq Whisper (fast, high quality, free 10h/month)
Fallback: Browser Whisper.js (handled by frontend, no server dependency)
"""

from __future__ import annotations

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
