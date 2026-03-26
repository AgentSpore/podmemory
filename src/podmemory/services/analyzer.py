"""LLM-based video analysis: transcript → structured knowledge extraction."""

from __future__ import annotations

import json
import re

import httpx

from ..core.config import settings
from .transcript import Transcript

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


class AnalysisError(Exception):
    pass


async def analyze_transcript(transcript: Transcript, model: str = "") -> dict:
    used_model = model or settings.llm_model

    if not settings.openrouter_api_key:
        raise AnalysisError(
            "No API key configured. Set PM_OPENROUTER_API_KEY or OPENROUTER_API_KEY."
        )

    # Truncate very long transcripts (free models have token limits)
    text = transcript.text
    if len(text) > 15000:
        text = text[:15000] + "\n\n[Transcript truncated at 15000 characters]"

    # Estimate duration from segments or text length
    duration_sec = 0
    if transcript.segments:
        last = transcript.segments[-1]
        duration_sec = int(last.get("start", 0)) + 10
    if not duration_sec:
        duration_sec = max(10, len(transcript.text) // 15)  # ~15 chars per second speech

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

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": used_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4000,
                },
            )

            if resp.status_code != 200:
                err = resp.json().get("error", {}).get("message", resp.text[:200])
                raise AnalysisError(f"LLM error ({resp.status_code}): {err}")

            content = resp.json()["choices"][0]["message"]["content"]

            # Clean markdown fences if present
            content = re.sub(r"```json?\s*", "", content)
            content = re.sub(r"```", "", content)
            content = content.strip()

            result = json.loads(content)

            # Validate required fields
            for field in ["title", "tldr", "key_insights", "flashcards"]:
                if field not in result:
                    raise AnalysisError(f"LLM response missing field: {field}")

            if not result["flashcards"]:
                raise AnalysisError("LLM returned empty flashcards")

            result["model_used"] = used_model
            result["transcript_source"] = transcript.source
            result["transcript_language"] = transcript.language
            result["transcript_length"] = len(transcript.text)

            return result

    except AnalysisError:
        raise
    except json.JSONDecodeError as e:
        raise AnalysisError(f"Failed to parse LLM response as JSON: {e}")
    except httpx.TimeoutException:
        raise AnalysisError("LLM request timed out. Try again or use a different model.")
    except Exception as e:
        raise AnalysisError(f"Failed to connect to AI: {e}")
