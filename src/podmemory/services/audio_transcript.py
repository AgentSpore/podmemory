"""Audio transcription utilities.

Transcription happens CLIENT-SIDE via Whisper.js in the browser.
This module only provides URL type detection.
No server-side transcription — full privacy, no API keys needed.
"""

from urllib.parse import urlparse

MEDIA_EXTENSIONS = {
    ".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".opus",
    ".flac", ".aac", ".wma", ".avi", ".mkv", ".mov",
}


def is_direct_media_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in MEDIA_EXTENSIONS)
