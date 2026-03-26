"""LLM-based video analysis: transcript → structured knowledge extraction."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict

import httpx

from ..core.config import settings
from .transcript import Transcript

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are PodMemory — an expert knowledge extraction AI.
Your job: take a video transcript and extract maximum learning value from it.

Rules:
- LANGUAGE: Detect the language of the transcript. Write ALL output in that SAME language.
- Be precise, analytical, and thorough
- Flashcards should test understanding, not just recall
- Timestamps MUST be within the actual video duration (provided below). Do NOT invent timestamps beyond the video length. If video is short (under 60s), use fewer timestamps or omit them.
- Action items should be concrete and actionable
- Scale output to content length: short transcripts get fewer insights/cards, long ones get more

Return ONLY valid JSON (no markdown fences):
{
  "title": "Brief descriptive title for this content",
  "tldr": "2-3 sentence summary of the core message",
  "key_insights": [
    "Insight 1 — the most important takeaway",
    "Insight 2",
    "..."
  ],
  "flashcards": [
    {"q": "Question testing understanding", "a": "Clear, concise answer"},
    {"q": "...", "a": "..."}
  ],
  "timestamps": [
    {"time": "0:00", "label": "Introduction"},
    {"time": "2:30", "label": "Key concept explained"},
    {"time": "..."}
  ],
  "action_items": [
    "Concrete thing you can do right now based on this content",
    "..."
  ],
  "tags": ["topic1", "topic2", "topic3"],
  "difficulty": "beginner|intermediate|advanced"
}

Generate 10-20 flashcards, 5-10 key insights, relevant timestamps, and 3-5 action items.
Quality over quantity — each flashcard should test real understanding."""

# Fallback models when primary fails (free tier on OpenRouter)
FALLBACK_MODELS = [
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-8b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

# In-memory LRU cache: key → (result, timestamp)
_analysis_cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
_CACHE_MAX = 50
_CACHE_TTL = 3600  # 1 hour


class AnalysisError(Exception):
    pass


def _cache_key(text: str, model: str) -> str:
    """Hash of transcript text + model for cache lookup."""
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
    """Truncate long transcripts preserving beginning + end at sentence boundaries."""
    if len(text) <= max_chars:
        return text

    # Keep 70% from beginning, 30% from end
    head_budget = int(max_chars * 0.7)
    tail_budget = max_chars - head_budget - 100  # 100 chars for separator

    # Cut at sentence boundary (., !, ?, newline)
    head = text[:head_budget]
    last_sentence = max(
        head.rfind(". "), head.rfind("! "), head.rfind("? "), head.rfind("\n")
    )
    if last_sentence > head_budget * 0.5:
        head = head[: last_sentence + 1]

    tail = text[-tail_budget:]
    first_sentence = min(
        (tail.find(". ") if tail.find(". ") >= 0 else 9999),
        (tail.find("! ") if tail.find("! ") >= 0 else 9999),
        (tail.find("? ") if tail.find("? ") >= 0 else 9999),
        (tail.find("\n") if tail.find("\n") >= 0 else 9999),
    )
    if first_sentence < tail_budget * 0.5:
        tail = tail[first_sentence + 2 :]

    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n[... {omitted} characters omitted ...]\n\n{tail}"


async def _call_llm(model: str, messages: list[dict]) -> str:
    """Call OpenRouter LLM API. Returns raw content string."""
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


def _parse_llm_response(content: str, model: str, transcript: Transcript) -> dict:
    """Parse and validate LLM JSON response."""
    content = re.sub(r"```json?\s*", "", content)
    content = re.sub(r"```", "", content)
    content = content.strip()

    result = json.loads(content)

    for field in ["title", "tldr", "key_insights", "flashcards"]:
        if field not in result:
            raise AnalysisError(f"LLM response missing field: {field}")

    if not result["flashcards"]:
        raise AnalysisError("LLM returned empty flashcards")

    result["model_used"] = model
    result["transcript_source"] = transcript.source
    result["transcript_language"] = transcript.language
    result["transcript_length"] = len(transcript.text)

    return result


async def analyze_transcript(transcript: Transcript, model: str = "") -> dict:
    used_model = model or settings.llm_model

    if not settings.openrouter_api_key:
        raise AnalysisError(
            "No API key configured. Set PM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
        )

    # Check cache
    ck = _cache_key(transcript.text, used_model)
    cached = _cache_get(ck)
    if cached:
        logger.info("Cache hit for model=%s", used_model)
        return cached

    # Smart truncation (preserves beginning + end)
    text = smart_truncate(transcript.text)

    # Estimate duration
    duration_sec = 0
    if transcript.segments:
        last = transcript.segments[-1]
        duration_sec = int(last.get("start", 0)) + 10
    if not duration_sec:
        duration_sec = max(10, len(transcript.text) // 15)

    duration_str = f"{duration_sec // 60}:{duration_sec % 60:02d}"

    user_prompt = f"""Analyze this video transcript and extract knowledge:

Transcript source: {transcript.source}
Language: {transcript.language}
Video duration: {duration_str} ({duration_sec} seconds)

IMPORTANT: All timestamps must be between 0:00 and {duration_str}. Do not generate timestamps beyond the video duration.

---
{text}
---

Generate a comprehensive analysis with flashcards for spaced repetition."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Try primary model, then fallbacks
    models_to_try = [used_model] + [m for m in FALLBACK_MODELS if m != used_model]

    last_error = None
    for i, m in enumerate(models_to_try):
        try:
            if i > 0:
                logger.warning("Fallback to model: %s", m)

            content = await _call_llm(m, messages)
            result = _parse_llm_response(content, m, transcript)

            # Cache successful result
            _cache_put(ck, result)
            return result

        except AnalysisError as e:
            last_error = e
            if i < len(models_to_try) - 1:
                logger.warning("Model %s failed: %s — trying next", m, e)
                continue
            raise
        except json.JSONDecodeError as e:
            last_error = AnalysisError(f"Failed to parse LLM response as JSON: {e}")
            if i < len(models_to_try) - 1:
                logger.warning("Model %s returned invalid JSON — trying next", m)
                continue
            raise last_error
        except httpx.TimeoutException:
            last_error = AnalysisError("LLM request timed out.")
            if i < len(models_to_try) - 1:
                logger.warning("Model %s timed out — trying next", m)
                continue
            raise last_error
        except Exception as e:
            last_error = AnalysisError(f"Failed to connect to AI: {e}")
            if i < len(models_to_try) - 1:
                continue
            raise last_error

    raise last_error or AnalysisError("All models failed")
