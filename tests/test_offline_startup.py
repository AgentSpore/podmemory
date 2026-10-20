"""Verify startup and provider boundaries with genuinely absent credentials."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from podmemory.services import audio_transcript


def test_application_starts_without_provider_credentials(tmp_path):
    """Import the real application outside conftest's synthetic environment."""
    script = """
from unittest.mock import patch
from fastapi.testclient import TestClient

with patch('httpx.AsyncClient.send', side_effect=AssertionError('Outbound HTTP')):
    from podmemory.core.config import settings
    from podmemory.main import app

    assert settings.groq_api_key == settings.openrouter_api_key == ''
    with TestClient(app) as client:
        assert client.get('/health').json()['status'] == 'ok'
        assert client.get('/').status_code == 200
        response = client.post('/api/export/anki', json={
            'title': 'Offline deck', 'flashcards': [{'q': 'Question?', 'a': 'Answer.'}]
        })
        assert response.status_code == 200
        assert response.content.startswith(b'PK')
        response = client.post('/api/analyze', json={'text': 'Study material.'})
        assert response.status_code == 502
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={
            "PATH": os.defpath,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "GROQ_API_KEY": "",
            "PM_GROQ_API_KEY": "",
            "OPENROUTER_API_KEY": "",
            "PM_OPENROUTER_API_KEY": "",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_missing_groq_key_rejects_before_tempfile_or_http(monkeypatch):
    monkeypatch.setattr(audio_transcript.settings, "groq_api_key", "")
    tempfile = Mock(side_effect=AssertionError("Temporary file created"))
    http_client = Mock(side_effect=AssertionError("HTTP client created"))
    monkeypatch.setattr(audio_transcript.tempfile, "mkstemp", tempfile)
    monkeypatch.setattr(httpx, "AsyncClient", http_client)

    with pytest.raises(RuntimeError, match="^NO_GROQ_KEY$"):
        await audio_transcript.transcribe_with_groq(b"synthetic audio")

    tempfile.assert_not_called()
    http_client.assert_not_called()


def test_missing_groq_key_returns_existing_unavailable_response(client, monkeypatch):
    monkeypatch.setattr(audio_transcript.settings, "groq_api_key", "")
    http_client = Mock(side_effect=RuntimeError("Unexpected provider call"))
    monkeypatch.setattr(httpx, "AsyncClient", http_client)
    response = client.post(
        "/api/transcribe",
        files={"file": ("audio.mp3", b"synthetic audio", "audio/mpeg")},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "GROQ_UNAVAILABLE"}
    http_client.assert_not_called()
