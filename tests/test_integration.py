"""Integration tests for PodMemory API endpoints."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from podmemory.services.transcript import detect_platform, Transcript


# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

_MOCK_ANALYSIS = {
    "title": "Test",
    "tldr": "Test summary",
    "key_insights": ["insight"],
    "flashcards": [{"q": "Q", "a": "A"}],
    "timestamps": [],
    "action_items": [],
    "tags": ["test"],
    "difficulty": "beginner",
    "model_used": "test",
    "transcript_source": "user_paste",
    "transcript_language": "en",
    "transcript_length": 10,
}

_MOCK_TRANSCRIPT = Transcript(
    text="Hello this is a test transcript for integration testing.",
    segments=[{"start": 0.0, "text": "Hello this is a test transcript"}],
    source="youtube_manual",
    language="en",
)


def _parse_sse_events(text: str) -> list[dict]:
    """Parse SSE response text into list of JSON payloads."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            raw = line[len("data:"):].strip()
            if raw:
                events.append(json.loads(raw))
    return events


# ===========================================================================
# 1. SSE /api/analyze-stream — text input
# ===========================================================================


class TestAnalyzeStreamText:
    def test_streams_stages_for_text(self, client: TestClient):
        """POST with text should stream: text_ready -> analyzing -> done."""
        with patch(
            "podmemory.api.analyze.analyze_transcript",
            new_callable=AsyncMock,
            return_value=_MOCK_ANALYSIS,
        ):
            resp = client.post("/api/analyze-stream", json={"text": "Some transcript text."})

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)

        stages = [e["stage"] for e in events]
        assert "text_ready" in stages
        assert "analyzing" in stages
        assert "done" in stages

        # done event must carry analysis data
        done_event = next(e for e in events if e["stage"] == "done")
        assert done_event["progress"] == 100
        assert "data" in done_event
        assert done_event["data"]["title"] == "Test"

    def test_progress_increases(self, client: TestClient):
        """Progress values should be monotonically non-decreasing."""
        with patch(
            "podmemory.api.analyze.analyze_transcript",
            new_callable=AsyncMock,
            return_value=_MOCK_ANALYSIS,
        ):
            resp = client.post("/api/analyze-stream", json={"text": "Test text"})

        events = _parse_sse_events(resp.text)
        progresses = [e["progress"] for e in events]
        assert progresses == sorted(progresses)


# ===========================================================================
# 2. SSE /api/analyze-stream — YouTube URL
# ===========================================================================


class TestAnalyzeStreamYouTube:
    def test_streams_stages_for_youtube(self, client: TestClient):
        """YouTube URL should stream: fetching_subtitles -> subtitles_ready -> analyzing -> done."""
        with (
            patch(
                "podmemory.api.analyze.get_youtube_transcript",
                new_callable=AsyncMock,
                return_value=_MOCK_TRANSCRIPT,
            ),
            patch(
                "podmemory.api.analyze.analyze_transcript",
                new_callable=AsyncMock,
                return_value=_MOCK_ANALYSIS,
            ),
        ):
            resp = client.post(
                "/api/analyze-stream",
                json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            )

        assert resp.status_code == 200
        events = _parse_sse_events(resp.text)
        stages = [e["stage"] for e in events]
        assert "fetching_subtitles" in stages
        assert "subtitles_ready" in stages
        assert "analyzing" in stages
        assert "done" in stages

    def test_youtube_no_subtitles_error(self, client: TestClient):
        """When YouTube has no subtitles, stream should emit error stage."""
        with patch(
            "podmemory.api.analyze.get_youtube_transcript",
            new_callable=AsyncMock,
            side_effect=ValueError("No subtitles found"),
        ):
            resp = client.post(
                "/api/analyze-stream",
                json={"url": "https://youtu.be/dQw4w9WgXcQ"},
            )

        events = _parse_sse_events(resp.text)
        stages = [e["stage"] for e in events]
        assert "fetching_subtitles" in stages
        assert "error" in stages
        error_event = next(e for e in events if e["stage"] == "error")
        assert "NO_SUBTITLES" in error_event["message"]


# ===========================================================================
# 3. SSE error handling — empty body
# ===========================================================================


class TestAnalyzeStreamErrors:
    def test_empty_body_emits_error_stage(self, client: TestClient):
        """POST without url or text should stream an error event."""
        resp = client.post("/api/analyze-stream", json={})
        events = _parse_sse_events(resp.text)
        stages = [e["stage"] for e in events]
        assert "error" in stages

    def test_empty_strings_emits_error_stage(self, client: TestClient):
        """Explicit empty strings should stream an error event."""
        resp = client.post("/api/analyze-stream", json={"url": "", "text": ""})
        events = _parse_sse_events(resp.text)
        stages = [e["stage"] for e in events]
        assert "error" in stages

    def test_unsupported_url_emits_error(self, client: TestClient):
        """URL from unsupported platform should emit error."""
        resp = client.post(
            "/api/analyze-stream",
            json={"url": "https://example.com/video/12345"},
        )
        events = _parse_sse_events(resp.text)
        stages = [e["stage"] for e in events]
        assert "error" in stages


# ===========================================================================
# 4. Anki export /api/export/anki — happy path
# ===========================================================================


class TestAnkiExport:
    def test_export_returns_apkg(self, client: TestClient):
        """Valid flashcards should produce a non-empty .apkg file."""
        resp = client.post(
            "/api/export/anki",
            json={
                "title": "My Deck",
                "flashcards": [
                    {"q": "What is Python?", "a": "A programming language"},
                    {"q": "What is FastAPI?", "a": "A web framework"},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert "content-disposition" in resp.headers
        assert len(resp.content) > 0
        # .apkg is a zip — starts with PK magic bytes
        assert resp.content[:2] == b"PK"

    def test_export_filename_in_disposition(self, client: TestClient):
        """Content-Disposition should contain the title as filename."""
        resp = client.post(
            "/api/export/anki",
            json={
                "title": "Test Deck",
                "flashcards": [{"q": "Q", "a": "A"}],
            },
        )
        assert resp.status_code == 200
        disp = resp.headers["content-disposition"]
        assert "Test Deck.apkg" in disp


# ===========================================================================
# 5. Anki export — cyrillic title (UTF-8)
# ===========================================================================


class TestAnkiExportCyrillic:
    def test_cyrillic_title_no_unicode_error(self, client: TestClient):
        """Russian title must not cause UnicodeEncodeError in headers."""
        resp = client.post(
            "/api/export/anki",
            json={
                "title": "Мой набор карточек",
                "flashcards": [
                    {"q": "Что такое Python?", "a": "Язык программирования"},
                ],
            },
        )
        assert resp.status_code == 200
        disp = resp.headers["content-disposition"]
        # RFC 5987: filename* should contain UTF-8 encoded title
        assert "filename*=UTF-8''" in disp
        # URL-encoded cyrillic characters should be present
        assert "%D0%9C" in disp  # "М" in URL-encoded UTF-8

    def test_cyrillic_ascii_fallback_filename(self, client: TestClient):
        """ASCII fallback filename should strip non-ASCII chars gracefully."""
        resp = client.post(
            "/api/export/anki",
            json={
                "title": "Тест",
                "flashcards": [{"q": "Q", "a": "A"}],
            },
        )
        assert resp.status_code == 200
        disp = resp.headers["content-disposition"]
        # ASCII fallback should be "podmemory.apkg" since "Тест" has no ASCII chars
        assert "podmemory.apkg" in disp


# ===========================================================================
# 6. Anki export validation — error cases
# ===========================================================================


class TestAnkiExportValidation:
    def test_empty_flashcards_returns_400(self, client: TestClient):
        """Empty flashcards list should be rejected with 400."""
        resp = client.post(
            "/api/export/anki",
            json={"title": "Empty", "flashcards": []},
        )
        assert resp.status_code == 400

    def test_invalid_structure_returns_422(self, client: TestClient):
        """Invalid flashcard structure should fail Pydantic validation (422)."""
        resp = client.post(
            "/api/export/anki",
            json={
                "title": "Bad",
                "flashcards": [{"wrong_field": "value"}],
            },
        )
        assert resp.status_code == 422

    def test_missing_flashcards_field_returns_422(self, client: TestClient):
        """Missing required 'flashcards' field should fail validation."""
        resp = client.post(
            "/api/export/anki",
            json={"title": "No cards"},
        )
        assert resp.status_code == 422


# ===========================================================================
# 7. YouTube audio — video ID validation
# ===========================================================================


class TestYouTubeAudioValidation:
    def test_valid_video_id_accepted(self, client: TestClient):
        """Valid 11-char video ID should pass validation (mock download)."""
        with patch(
            "podmemory.api.analyze.download_youtube_audio",
            new_callable=AsyncMock,
        ) as mock_dl:
            import tempfile
            from pathlib import Path

            # Create a real temp file so FileResponse can read it
            tmp = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
            tmp.write(b"\x00" * 100)
            tmp.close()
            mock_dl.return_value = Path(tmp.name)

            try:
                resp = client.get("/api/youtube-audio/dQw4w9WgXcQ")
                # Should succeed (200) or at least not be 400
                assert resp.status_code == 200
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    def test_path_traversal_rejected(self, client: TestClient):
        """Path traversal attempts should be rejected."""
        resp = client.get("/api/youtube-audio/../etc/passwd")
        # FastAPI route won't match (404) or regex rejects (400)
        assert resp.status_code in (400, 404)

    def test_short_id_rejected(self, client: TestClient):
        """Too-short video ID should fail validation."""
        resp = client.get("/api/youtube-audio/abc")
        assert resp.status_code == 400

    def test_long_id_rejected(self, client: TestClient):
        """Too-long video ID should fail validation."""
        resp = client.get("/api/youtube-audio/abcdefghijklm")
        assert resp.status_code == 400

    def test_special_chars_rejected(self, client: TestClient):
        """Video ID with special characters should fail validation."""
        resp = client.get("/api/youtube-audio/abc!@#$%^&*(")
        # URL encoding may vary but should not pass the regex
        assert resp.status_code in (400, 404)


# ===========================================================================
# 8. detect_platform — all supported platforms
# ===========================================================================


class TestDetectPlatform:
    @pytest.mark.parametrize(
        "url, expected",
        [
            # YouTube — various formats
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
            ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
            ("https://youtube.com/shorts/dQw4w9WgXcQ", "youtube"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
            # VK Video
            ("https://vk.com/video-12345_67890", "vk"),
            ("https://vkvideo.ru/video-12345_67890", "vk"),
            ("https://vk.com/clip-12345_67890", "vk"),
            # Rutube
            ("https://rutube.ru/video/abc123def456/", "rutube"),
            # Dzen
            ("https://dzen.ru/video/watch/abc123", "dzen"),
            ("https://dzen.ru/watch/abc123", "dzen"),
            # OK.ru
            ("https://ok.ru/video/123456789", "ok"),
            # Yandex Music
            ("https://music.yandex.ru/album/123/track/456", "yandex_music"),
            # Twitch
            ("https://www.twitch.tv/videos/123456", "twitch"),
            ("https://clips.twitch.tv/SomeClipName", "twitch"),
            # Vimeo
            ("https://vimeo.com/123456789", "vimeo"),
            # SoundCloud
            ("https://soundcloud.com/artist/track-name", "soundcloud"),
            # Telegram
            ("https://t.me/channel/123", "telegram"),
            # Rumble
            ("https://rumble.com/v123-title.html", "rumble"),
            # Apple Podcasts
            ("https://podcasts.apple.com/us/podcast/ep-1/id123", "apple_podcasts"),
            # Bilibili
            ("https://www.bilibili.com/video/BV1xx411c7mD", "bilibili"),
            # Coub
            ("https://coub.com/view/abc123", "coub"),
            # Unsupported
            ("https://example.com/video/123", None),
            ("https://tiktok.com/@user/video/123", None),
            ("", None),
            ("not-a-url", None),
        ],
    )
    def test_detect_platform(self, url: str, expected: str | None):
        assert detect_platform(url) == expected
