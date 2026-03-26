"""Extract transcripts from video URLs.

Supported platforms:
- YouTube (via youtube-transcript-api — instant, no API key needed)
- 1000+ sites (via yt-dlp + Groq Whisper — needs GROQ_API_KEY)
- Manual paste fallback
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Transcript:
    text: str
    segments: list[dict]  # [{"start": float, "text": str}, ...]
    source: str  # "youtube_auto", "youtube_manual", "user_paste"
    language: str


def extract_youtube_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def is_youtube_url(url: str) -> bool:
    return extract_youtube_id(url) is not None


async def get_youtube_transcript(url: str) -> Transcript:
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL. Check the link and try again.")

    try:
        yt = YouTubeTranscriptApi()
        transcript_list = yt.list(video_id)
    except Exception as e:
        msg = str(e)
        if "unplayable" in msg.lower() or "unavailable" in msg.lower() or "not available" in msg.lower():
            raise ValueError(
                "This video is unavailable or restricted. "
                "Try uploading the audio/video file instead."
            )
        if "no transcript" in msg.lower() or "could not retrieve" in msg.lower():
            raise ValueError(
                "No subtitles found for this video. "
                "Try uploading the audio/video file — it will be transcribed in your browser."
            )
        raise ValueError(f"Could not get transcript: {msg[:150]}")

    # Prefer manual transcripts, then auto-generated
    transcript_obj = None
    source = "youtube_auto"
    lang = "en"

    try:
        for t in transcript_list:
            if not t.is_generated:
                transcript_obj = t
                source = "youtube_manual"
                lang = t.language_code
                break
    except Exception:
        pass

    if transcript_obj is None:
        try:
            for t in transcript_list:
                transcript_obj = t
                source = "youtube_auto"
                lang = t.language_code
                break
        except Exception:
            pass

    if transcript_obj is None:
        raise ValueError(
            "No subtitles found for this video. "
            "Try uploading the audio/video file instead."
        )

    snippets = transcript_obj.fetch()
    segments = []
    full_text_parts = []

    for s in snippets:
        segments.append({"start": s.start, "text": s.text})
        full_text_parts.append(s.text)

    return Transcript(
        text=" ".join(full_text_parts),
        segments=segments,
        source=source,
        language=lang,
    )


def from_user_paste(text: str, language: str = "unknown") -> Transcript:
    return Transcript(
        text=text.strip(),
        segments=[],
        source="user_paste",
        language=language,
    )
