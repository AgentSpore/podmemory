"""LLM-based content analysis: video/article/book → structured knowledge extraction."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict

import httpx
from loguru import logger

from ..core.config import settings
from .transcript import Transcript

_VIDEO_PROMPT = """You are PodMemory — an expert knowledge extraction AI.
Your job: take a video transcript and extract maximum learning value from it.

Rules:
- LANGUAGE: Detect the language of the content. Write ALL output in that SAME language.
- Be precise, analytical, and thorough
- Flashcards should test understanding, not just recall
- Timestamps MUST be within the actual video duration (provided below). Do NOT invent timestamps beyond the video length.
- Action items should be concrete and actionable
- Scale output to content length: short content gets fewer insights/cards, long ones get more

Return ONLY valid JSON (no markdown fences):
{
  "title": "Brief descriptive title",
  "tldr": "2-3 sentence summary of the core message",
  "key_insights": ["Insight 1", "Insight 2", "..."],
  "flashcards": [{"q": "Question testing understanding", "a": "Clear answer"}, ...],
  "timestamps": [{"time": "0:00", "label": "Introduction"}, {"time": "2:30", "label": "Key concept"}, ...],
  "action_items": ["Concrete actionable step", "..."],
  "tags": ["topic1", "topic2"],
  "difficulty": "beginner|intermediate|advanced"
}

Generate 10-20 flashcards, 5-10 key insights, relevant timestamps, and 3-5 action items.
Quality over quantity — each flashcard should test real understanding."""

_TEXT_PROMPT = """You are PodMemory — an expert knowledge extraction AI.
Your job: take an article or book text and extract maximum learning value from it.

Rules:
- LANGUAGE: Detect the language of the content. Write ALL output in that SAME language.
- Be precise, analytical, and thorough
- Flashcards should test understanding, not just recall
- Do NOT generate timestamps — this is text content, not video
- Extract key vocabulary terms with clear definitions
- Pick the most memorable and insightful quotes from the text
- Action items should be concrete and actionable
- Scale output to content length: short articles get fewer insights/cards, books get more

Return ONLY valid JSON (no markdown fences):
{
  "title": "Brief descriptive title",
  "tldr": "2-3 sentence summary of the core message",
  "key_insights": ["Insight 1", "Insight 2", "..."],
  "flashcards": [{"q": "Question testing understanding", "a": "Clear answer"}, ...],
  "vocabulary": [{"term": "Key concept", "definition": "Clear explanation"}, ...],
  "quotes": ["Most memorable quote from the text", "..."],
  "action_items": ["Concrete actionable step", "..."],
  "tags": ["topic1", "topic2"],
  "difficulty": "beginner|intermediate|advanced"
}

Generate 10-20 flashcards, 5-10 key insights, 5-10 vocabulary terms, 3-5 quotes, and 3-5 action items.
Quality over quantity — each flashcard should test real understanding."""

FALLBACK_MODELS = [
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-8b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

_analysis_cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
_CACHE_MAX = 50
_CACHE_TTL = 3600


class AnalysisError(Exception):
    pass


def _cache_key(text: str, model: str) -> str:
    h = hashlib.md5(f"{model}:{text[:5000]}".encode()).hexdigest()
    return h


def _cache_get(key: str) -> dict | None:
    if key in _analysis_cache:
        result, ts = _analysis_cache[key]
        if time.time() - ts < _CACHE_TTL:
            _analysis_cache.move_to_end(key)
            return result
        del _analysis_cache[key]
    return None


def _cache_put(key: str, result: dict):
    _analysis_cache[key] = (result, time.time())
    if len(_analysis_cache) > _CACHE_MAX:
        _analysis_cache.popitem(last=False)


def smart_truncate(text: str, max_chars: int = 14000) -> str:
    """Truncate long content preserving beginning + end at sentence boundaries."""
    if len(text) <= max_chars:
        return text

    head_budget = int(max_chars * 0.7)
    tail_budget = max_chars - head_budget - 100

    head = text[:head_budget]
    last_sentence = max(
        head.rfind(". "), head.rfind("! "), head.rfind("? "), head.rfind("\n")
    )
    if last_sentence > head_budget * 0.5:
        head = head[: last_sentence + 1]

    tail = text[-tail_budget:]
    sentence_positions = [
        pos for pos in (tail.find(". "), tail.find("! "), tail.find("? "), tail.find("\n"))
        if pos >= 0
    ]
    first_sentence = min(sentence_positions) if sentence_positions else tail_budget
    if first_sentence < tail_budget * 0.5:
        tail = tail[first_sentence + 2 :]

    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n[... {omitted} characters omitted ...]\n\n{tail}"


async def _call_llm(model: str, messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 4000,
            },
        )

        if resp.status_code != 200:
            err = resp.json().get("error", {}).get("message", resp.text[:200])
            raise AnalysisError(f"LLM error ({resp.status_code}): {err}")

        return resp.json()["choices"][0]["message"]["content"]


def _parse_llm_response(content: str, model: str, transcript: Transcript, source_type: str) -> dict:
    content = re.sub(r"```json?\s*", "", content)
    content = re.sub(r"```", "", content)
    content = content.strip()

    result = json.loads(content)

    for field in ["title", "tldr", "key_insights", "flashcards"]:
        if field not in result:
            raise AnalysisError(f"LLM response missing field: {field}")

    if not result["flashcards"]:
        raise AnalysisError("LLM returned empty flashcards")

    result.setdefault("timestamps", [])
    result.setdefault("vocabulary", [])
    result.setdefault("quotes", [])
    result.setdefault("action_items", [])
    result.setdefault("tags", [])
    result.setdefault("difficulty", "intermediate")

    result["source_type"] = source_type
    result["model_used"] = model
    result["transcript_source"] = transcript.source
    result["transcript_language"] = transcript.language
    result["transcript_length"] = len(transcript.text)

    return result


def _build_video_prompt(transcript: Transcript, text: str) -> tuple[str, str]:
    """Build system + user prompts for video content."""
    duration_sec = 0
    if transcript.segments:
        last = transcript.segments[-1]
        duration_sec = int(last.get("start", 0)) + 10
    if not duration_sec:
        duration_sec = max(10, len(transcript.text) // 15)

    duration_str = f"{duration_sec // 60}:{duration_sec % 60:02d}"

    user_prompt = f"""Analyze this video transcript and extract knowledge:

Source: {transcript.source}
Language: {transcript.language}
Video duration: {duration_str} ({duration_sec} seconds)

IMPORTANT: All timestamps must be between 0:00 and {duration_str}.

---
{text}
---

Generate a comprehensive analysis with flashcards for spaced repetition."""

    return _VIDEO_PROMPT, user_prompt


def _build_text_prompt(transcript: Transcript, text: str) -> tuple[str, str]:
    """Build system + user prompts for article/book content."""
    word_count = len(text.split())

    user_prompt = f"""Analyze this text and extract knowledge:

Source: {transcript.source}
Language: {transcript.language}
Length: ~{word_count} words

---
{text}
---

Generate a comprehensive analysis with flashcards, vocabulary terms, and memorable quotes for retention."""

    return _TEXT_PROMPT, user_prompt


async def analyze_transcript(transcript: Transcript, model: str = "", source_type: str = "") -> dict:
    """Analyze content. source_type: 'video' (default), 'article', 'pdf', 'text'."""
    used_model = model or settings.llm_model

    if not settings.openrouter_api_key:
        raise AnalysisError(
            "No API key configured. Set PM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
        )

    if not source_type:
        source_type = "video" if transcript.source in (
            "youtube_auto", "youtube_manual", "groq_whisper",
        ) or "groq_whisper" in transcript.source or "subtitles" in transcript.source else "text"

    ck = _cache_key(transcript.text, used_model)
    cached = _cache_get(ck)
    if cached:
        logger.info("Cache hit for model={}", used_model)
        return cached

    text = smart_truncate(transcript.text)

    if source_type == "video":
        system_prompt, user_prompt = _build_video_prompt(transcript, text)
    else:
        system_prompt, user_prompt = _build_text_prompt(transcript, text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    models_to_try = [used_model] + [m for m in FALLBACK_MODELS if m != used_model]

    last_error = None
    for i, m in enumerate(models_to_try):
        try:
            if i > 0:
                logger.warning("Fallback to model: {}", m)

            content = await _call_llm(m, messages)
            result = _parse_llm_response(content, m, transcript, source_type)

            _cache_put(ck, result)
            return result

        except (AnalysisError, json.JSONDecodeError, httpx.TimeoutException, Exception) as e:
            if isinstance(e, AnalysisError):
                last_error = e
            elif isinstance(e, json.JSONDecodeError):
                last_error = AnalysisError(f"Failed to parse LLM response as JSON: {e}")
            elif isinstance(e, httpx.TimeoutException):
                last_error = AnalysisError("LLM request timed out.")
            else:
                last_error = AnalysisError(f"Failed to connect to AI: {e}")
            if i < len(models_to_try) - 1:
                logger.warning("Model {} failed: {} — trying next", m, last_error)
                continue
            raise last_error

    raise last_error or AnalysisError("All models failed")
