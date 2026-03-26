"""Smoke tests for PodMemory service."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from podmemory.services.transcript import extract_youtube_id, from_user_paste


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------


def test_app_starts(client: TestClient):
    """Application boots without errors."""
    assert client.app is not None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["app"] == "PodMemory"


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------


def test_config_returns_models_list(client: TestClient):
    """Config endpoint returns models as a list (may be empty if network unavailable)."""
    with patch(
        "podmemory.api.analyze._fetch_free_models",
        new_callable=AsyncMock,
        return_value=[
            {"id": "test/model-a", "name": "Model A", "context_length": 4096},
            {"id": "test/model-b", "name": "Model B", "context_length": 8192},
        ],
    ):
        resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["models"], list)
    assert len(data["models"]) == 2
    assert "default_model" in data
    assert "has_api_key" in data


# ---------------------------------------------------------------------------
# POST /api/analyze — validation
# ---------------------------------------------------------------------------


def test_analyze_empty_body_returns_400(client: TestClient):
    """Empty request body must be rejected."""
    resp = client.post("/api/analyze", json={})
    assert resp.status_code == 400


def test_analyze_no_url_no_text_returns_400(client: TestClient):
    """Explicit empty strings for both url and text must be rejected."""
    resp = client.post("/api/analyze", json={"url": "", "text": ""})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/analyze — with text (mocked LLM)
# ---------------------------------------------------------------------------


_MOCK_LLM_RESPONSE = {
    "title": "Test Title",
    "tldr": "Test summary.",
    "key_insights": ["Insight 1"],
    "flashcards": [{"q": "What is X?", "a": "X is Y."}],
    "timestamps": [],
    "action_items": [],
    "tags": ["test"],
    "difficulty": "beginner",
}


def test_analyze_with_text_calls_llm(client: TestClient):
    """Text input should go through LLM analysis and return structured result."""
    with patch(
        "podmemory.api.analyze.analyze_transcript",
        new_callable=AsyncMock,
        return_value={
            **_MOCK_LLM_RESPONSE,
            "model_used": "test/model",
            "transcript_source": "user_paste",
            "transcript_language": "unknown",
            "transcript_length": 42,
        },
    ) as mock_analyze:
        resp = client.post("/api/analyze", json={"text": "Hello world this is a test transcript."})

    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test Title"
    assert isinstance(data["flashcards"], list)
    assert len(data["flashcards"]) > 0
    mock_analyze.assert_called_once()

    # Verify the transcript was created from user paste
    call_args = mock_analyze.call_args
    transcript_arg = call_args[0][0]
    assert transcript_arg.source == "user_paste"
    assert "Hello world" in transcript_arg.text


# ---------------------------------------------------------------------------
# extract_youtube_id — various URL formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected_id",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=120", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("not-a-youtube-url", None),
        ("https://example.com/watch?v=abc", None),
        ("", None),
    ],
)
def test_extract_youtube_id(url: str, expected_id: str | None):
    assert extract_youtube_id(url) == expected_id


# ---------------------------------------------------------------------------
# from_user_paste
# ---------------------------------------------------------------------------


def test_from_user_paste_strips_whitespace():
    t = from_user_paste("  hello world  \n\n")
    assert t.text == "hello world"
    assert t.source == "user_paste"
    assert t.segments == []


def test_from_user_paste_preserves_content():
    long_text = "Line 1\nLine 2\nLine 3"
    t = from_user_paste(long_text)
    assert t.text == long_text
    assert t.language == "unknown"


def test_from_user_paste_custom_language():
    t = from_user_paste("Bonjour le monde", language="fr")
    assert t.language == "fr"
