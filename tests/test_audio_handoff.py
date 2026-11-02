"""Exercise the real SSE-to-file handoff with synthetic bytes only."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from podmemory.api import analyze
from podmemory.main import app
from podmemory.schemas.analysis import AnalyzeRequest


AUDIO = b"synthetic local audio bytes for the browser handoff"
URL = "https://vimeo.com/123456789"


def events(response):
    assert response.status_code == 200
    return [
        json.loads(line[5:])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]


@pytest.fixture
def audio_source(tmp_path, monkeypatch):
    folder = tmp_path / "download"
    folder.mkdir()
    path = folder / "audio.mp3"
    path.write_bytes(AUDIO)
    monkeypatch.setattr(analyze, "_temp_audio", {})
    monkeypatch.setattr(analyze, "_audio_owners", {})
    monkeypatch.setattr(analyze, "fetch_subtitles", AsyncMock(return_value=None))
    monkeypatch.setattr(analyze, "download_audio", AsyncMock(return_value=path))
    monkeypatch.setattr(analyze, "analyze_transcript", AsyncMock())
    yield path
    for _, timer in analyze._temp_audio.values():
        timer.cancel()
    analyze._temp_audio.clear()
    for audio_id in list(analyze._audio_owners):
        analyze._release_temp_audio(audio_id)
    assert not analyze._audio_owners


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["GROQ_RATE_LIMIT", "Forbidden: synthetic"])
async def test_fallback_audio_survives_completed_stream(
    audio_source, monkeypatch, error
):
    transcribe = AsyncMock(side_effect=RuntimeError(error))
    monkeypatch.setattr(analyze, "transcribe_with_groq", transcribe)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        stages = events(await client.post("/api/analyze-stream", json={"url": URL}))
        fallback = next(e for e in stages if e["stage"] == "groq_fallback")
        response = await client.get(fallback["data"]["audio_url"])
        assert response.status_code == 200
        assert response.content == AUDIO
        assert response.headers["content-type"] == "audio/mpeg"
        assert (await client.get(fallback["data"]["audio_url"])).status_code == 404
    transcribe.assert_awaited_once_with(AUDIO, "vimeo_audio.m4a")
    analyze.analyze_transcript.assert_not_awaited()
    assert not audio_source.exists()
    assert not audio_source.parent.exists()


async def collect_stream():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return events(await client.post("/api/analyze-stream", json={"url": URL}))


@pytest.mark.asyncio
async def test_unclaimed_audio_expires_without_another_request(
    audio_source, monkeypatch
):
    cleaned = asyncio.Event()
    original = analyze._remove_audio_file

    def remove(path):
        removed = original(path)
        cleaned.set()
        return removed

    monkeypatch.setattr(analyze, "_TEMP_AUDIO_TTL", 0.001)
    monkeypatch.setattr(analyze, "_remove_audio_file", remove)
    audio_id = analyze._offer_temp_audio(audio_source)
    await asyncio.wait_for(cleaned.wait(), timeout=1)
    assert audio_id not in analyze._temp_audio
    assert not audio_source.exists()
    assert not audio_source.parent.exists()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get(f"/api/temp-audio/{audio_id}")).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "oversize", "unreadable"])
async def test_server_owned_audio_is_released(audio_source, monkeypatch, outcome):
    transcribe = AsyncMock(return_value={"text": "Synthetic study material."})
    if outcome == "error":
        transcribe.side_effect = RuntimeError("synthetic transcription failure")
    elif outcome == "oversize":
        monkeypatch.setattr(analyze, "_MAX_TRANSCRIPTION_BYTES", 8)
        audio_source.write_bytes(b"a" * 9)
    elif outcome == "unreadable":
        audio_source.unlink()
    monkeypatch.setattr(analyze, "transcribe_with_groq", transcribe)
    analyze.analyze_transcript.return_value = {"title": "Synthetic analysis"}
    stages = [event["stage"] for event in await collect_stream()]
    assert stages[-1] == ("done" if outcome == "success" else "error")
    assert "groq_fallback" not in stages
    assert not analyze._temp_audio
    assert not audio_source.exists()
    assert not audio_source.parent.exists()
    if outcome in ("oversize", "unreadable"):
        transcribe.assert_not_awaited()
    if outcome != "success":
        analyze.analyze_transcript.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["transcribing", "groq_fallback"])
async def test_generator_close_respects_audio_owner(audio_source, monkeypatch, stage):
    monkeypatch.setattr(
        analyze,
        "transcribe_with_groq",
        AsyncMock(side_effect=RuntimeError("GROQ_RATE_LIMIT")),
    )
    response = await analyze.analyze_video_stream(AnalyzeRequest(url=URL))
    iterator = response.body_iterator
    async for payload in iterator:
        event = json.loads(payload)
        if event["stage"] == stage:
            break
    else:
        pytest.fail(f"Expected stage {stage}")
    await iterator.aclose()
    if stage == "transcribing":
        assert not audio_source.exists()
        assert not audio_source.parent.exists()
        assert not analyze._temp_audio
    else:
        assert audio_source.read_bytes() == AUDIO
        audio_id = event["data"]["audio_url"].rsplit("/", 1)[-1]
        analyze._expire_temp_audio(audio_id)
        assert not audio_source.exists()
        assert not audio_source.parent.exists()


@pytest.mark.asyncio
async def test_cancelled_transcription_releases_audio(audio_source, monkeypatch):
    monkeypatch.setattr(
        analyze, "transcribe_with_groq", AsyncMock(side_effect=asyncio.CancelledError())
    )
    response = await analyze.analyze_video_stream(AnalyzeRequest(url=URL))
    with pytest.raises(asyncio.CancelledError):
        async for _ in response.body_iterator:
            pass
    assert not analyze._temp_audio
    assert not audio_source.exists()
    assert not audio_source.parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "range_header,status",
    [
        ("bytes=0-8", 206),
        ("bytes=9999-", 416),
        ("invalid", 400),
    ],
)
async def test_range_response_releases_claimed_audio(
    audio_source, range_header, status
):
    audio_id = analyze._offer_temp_audio(audio_source)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/temp-audio/{audio_id}", headers={"range": range_header}
        )
        assert response.status_code == status
        if status == 206:
            assert response.content == AUDIO[:9]
        assert (await client.get(f"/api/temp-audio/{audio_id}")).status_code == 404
    assert not audio_source.exists()
    assert not audio_source.parent.exists()


def audio_scope(audio_id):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": f"/api/temp-audio/{audio_id}",
        "query_string": b"",
        "headers": [],
        "server": ("test", 80),
        "client": ("test", 1234),
    }


async def receive_request():
    return {"type": "http.request", "body": b"", "more_body": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_failed_file_send_releases_audio(audio_source, failure):
    audio_id = analyze._offer_temp_audio(audio_source)

    async def send(message):
        if message["type"] == "http.response.body":
            raise failure("synthetic send failure")

    with pytest.raises(failure):
        await app(audio_scope(audio_id), receive_request, send)
    assert not analyze._temp_audio
    assert not audio_source.exists()
    assert not audio_source.parent.exists()


@pytest.mark.asyncio
async def test_expiry_cannot_delete_active_response(audio_source):
    audio_id = analyze._offer_temp_audio(audio_source)
    _, timer = analyze._temp_audio[audio_id]
    response = await analyze.get_temp_audio(audio_id)
    assert timer.cancelled()
    analyze._expire_temp_audio(audio_id)
    assert audio_source.read_bytes() == AUDIO
    sent = []

    async def send(message):
        sent.append(message)

    await response(audio_scope(audio_id), receive_request, send)
    assert b"".join(m.get("body", b"") for m in sent) == AUDIO
    assert not audio_source.exists()
    assert not audio_source.parent.exists()


@pytest.mark.asyncio
async def test_only_one_concurrent_claim_succeeds(audio_source):
    audio_id = analyze._offer_temp_audio(audio_source)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *[client.get(f"/api/temp-audio/{audio_id}") for _ in range(2)]
        )
    assert sorted(response.status_code for response in responses) == [200, 404]
    assert next(r for r in responses if r.status_code == 200).content == AUDIO
    assert not audio_source.exists()
    assert not audio_source.parent.exists()


@pytest.mark.asyncio
async def test_missing_file_returns_404_and_cancels_expiry(audio_source):
    audio_id = analyze._offer_temp_audio(audio_source)
    _, timer = analyze._temp_audio[audio_id]
    audio_source.unlink()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get(f"/api/temp-audio/{audio_id}")).status_code == 404
        assert (await client.get("/api/temp-audio/unknown")).status_code == 404
    assert timer.cancelled()
    assert not analyze._temp_audio
    assert not audio_source.parent.exists()
