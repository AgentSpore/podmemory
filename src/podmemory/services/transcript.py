"""Extract transcripts from video URLs.

Supported platforms:
- YouTube (via youtube-transcript-api — instant, no API key needed)
- VK Video, Rutube, and others (via yt-dlp + Groq Whisper)
- Manual paste fallback
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

# Noise patterns in YouTube auto-generated subtitles
_NOISE_PATTERNS = re.compile(
    r"\[(?:Music|Applause|Laughter|Cheering|Silence|Inaudible|Foreign)\]",
    re.IGNORECASE,
)
_MULTI_SPACES = re.compile(r"[ \t]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")


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


# Platforms supported via yt-dlp (audio download + Groq Whisper)
_SUPPORTED_PLATFORMS = {
    "vk": re.compile(r"(?:vk\.com/video|vkvideo\.ru|vk\.com/clip)", re.IGNORECASE),
    "rutube": re.compile(r"rutube\.ru/video/", re.IGNORECASE),
    "dzen": re.compile(r"dzen\.ru/(?:video|watch)", re.IGNORECASE),
    "ok": re.compile(r"ok\.ru/video/", re.IGNORECASE),
    "yandex_music": re.compile(r"music\.yandex\.(?:ru|com)/album/.+/track/", re.IGNORECASE),
    "twitch": re.compile(r"(?:twitch\.tv/|clips\.twitch\.tv/)", re.IGNORECASE),
    "vimeo": re.compile(r"vimeo\.com/\d+", re.IGNORECASE),
    "soundcloud": re.compile(r"soundcloud\.com/", re.IGNORECASE),
    "telegram": re.compile(r"t\.me/.+/\d+", re.IGNORECASE),
    "rumble": re.compile(r"rumble\.com/", re.IGNORECASE),
    "apple_podcasts": re.compile(r"podcasts\.apple\.com/", re.IGNORECASE),
    "bilibili": re.compile(r"bilibili\.com/video/", re.IGNORECASE),
    "coub": re.compile(r"coub\.com/view/", re.IGNORECASE),
}


def detect_platform(url: str) -> str | None:
    """Detect video platform from URL. Returns platform name or None."""
    if is_youtube_url(url):
        return "youtube"
    for name, pattern in _SUPPORTED_PLATFORMS.items():
        if pattern.search(url):
            return name
    return None



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
    except Exception as e:
        logger.warning("Error iterating manual transcripts for {}: {}", video_id, e)

    if transcript_obj is None:
        try:
            for t in transcript_list:
                transcript_obj = t
                source = "youtube_auto"
                lang = t.language_code
                break
        except Exception as e:
            logger.warning("Error iterating auto transcripts for {}: {}", video_id, e)

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

    raw_text = " ".join(full_text_parts)
    cleaned = clean_transcript(raw_text) if source == "youtube_auto" else raw_text

    return Transcript(
        text=cleaned,
        segments=segments,
        source=source,
        language=lang,
    )


def clean_transcript(text: str) -> str:
    """Remove noise from auto-generated subtitles."""
    text = _NOISE_PATTERNS.sub("", text)
    # Deduplicate consecutive identical phrases (common in auto-subs)
    words = text.split()
    cleaned = []
    prev_chunk = ""
    for i in range(0, len(words), 5):
        chunk = " ".join(words[i : i + 5])
        if chunk != prev_chunk:
            cleaned.append(chunk)
        prev_chunk = chunk
    text = " ".join(cleaned)
    text = _MULTI_SPACES.sub(" ", text)
    text = _MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()


def from_user_paste(text: str, language: str = "unknown") -> Transcript:
    return Transcript(
        text=text.strip(),
        segments=[],
        source="user_paste",
        language=language,
    )
