"""Download audio from YouTube when subtitles are unavailable.

Uses pytubefix (pure Python, no external tools).
Audio is served to the browser for client-side Whisper.js transcription.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


async def download_youtube_audio(video_id: str) -> Path:
    """Download audio from YouTube video using pytubefix.
    Returns path to audio file.
    """
    def _download():
        from pytubefix import YouTube

        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(url)

        # Get audio-only stream (smallest)
        stream = yt.streams.filter(only_audio=True).order_by("abr").first()
        if not stream:
            raise RuntimeError("No audio stream available for this video")

        tmp_dir = tempfile.mkdtemp(prefix="podmemory_")
        out_path = stream.download(output_path=tmp_dir, filename="audio")
        return Path(out_path)

    return await asyncio.to_thread(_download)
