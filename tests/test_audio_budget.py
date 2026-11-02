"""Low-budget actual HTTP/ASGI pressure regressions; no providers or large files."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from podmemory.api import analyze
from podmemory.main import app
from podmemory.schemas.analysis import AnalyzeRequest
from tests.test_audio_handoff import URL, audio_scope, events, receive_request


@pytest.fixture
def tiny_audio(tmp_path, monkeypatch):
    state = SimpleNamespace(paths=[], data=b"a" * 1024)

    async def download(_):
        folder = tmp_path / str(len(state.paths))
        folder.mkdir()
        path = folder / "audio.mp3"
        path.write_bytes(state.data)
        state.paths.append(path)
        return path

    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 2, raising=False)
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_BYTES", 2048, raising=False)
    monkeypatch.setattr(analyze, "_temp_audio", {})
    monkeypatch.setattr(analyze, "_audio_owners", {}, raising=False)
    monkeypatch.setattr(analyze, "fetch_subtitles", AsyncMock(return_value=None))
    monkeypatch.setattr(analyze, "download_audio", download)
    monkeypatch.setattr(
        analyze,
        "transcribe_with_groq",
        AsyncMock(side_effect=RuntimeError("GROQ_RATE_LIMIT")),
    )
    monkeypatch.setattr(analyze, "analyze_transcript", AsyncMock())
    yield state
    for _, timer in analyze._temp_audio.values():
        timer.cancel()
    analyze._temp_audio.clear()
    for audio_id in list(analyze._audio_owners):
        analyze._release_temp_audio(audio_id)
    assert not analyze._audio_owners


async def offer(client):
    return events(await client.post("/api/analyze-stream", json={"url": URL}))[-1]


def offered_url(event):
    assert event["stage"] == "groq_fallback"
    return event["data"]["audio_url"]


def assert_busy(event):
    assert event["stage"] == "error"
    assert "busy" in event["message"].lower()
    assert "audio_url" not in str(event)


@asynccontextmanager
async def hold_download(url):
    entered, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def send(message):
        if message["type"] == "http.response.body":
            entered.set()
            await release.wait()
        sent.append(message)

    task = asyncio.create_task(
        app(audio_scope(url.rsplit("/", 1)[-1]), receive_request, send)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        yield sent
    finally:
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["count", "bytes"])
async def test_pending_budget_refuses_and_cleans_new_file(
    tiny_audio, monkeypatch, limit
):
    if limit == "bytes":
        monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 8)
        monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_BYTES", 1500)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        urls = [offered_url(await offer(client))]
        if limit == "count":
            urls.append(offered_url(await offer(client)))
        assert_busy(await offer(client))
        assert not tiny_audio.paths[-1].exists()
        assert not tiny_audio.paths[-1].parent.exists()
        assert sum(p.exists() for p in tiny_audio.paths) == len(urls)
        for url in urls:
            response = await client.get(url)
            assert response.status_code == 200
            assert response.content == tiny_audio.data
        assert all(not p.exists() for p in tiny_audio.paths)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["count", "bytes"])
async def test_active_send_still_consumes_budget(tiny_audio, monkeypatch, limit):
    if limit == "bytes":
        monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 8)
        monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_BYTES", 1500)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        async with hold_download(url):
            if limit == "count":
                pending = offered_url(await offer(client))
            assert_busy(await offer(client))
            assert not tiny_audio.paths[-1].exists()
            assert tiny_audio.paths[0].exists()
        assert not tiny_audio.paths[0].exists()
        if limit == "count":
            assert (await client.get(pending)).status_code == 200
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).content == tiny_audio.data


@pytest.mark.asyncio
async def test_exact_byte_limit_then_one_byte_over_and_reuse(tiny_audio, monkeypatch):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_BYTES", 1024)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        tiny_audio.data = b"x"
        assert_busy(await offer(client))
        assert len(analyze._audio_owners) == 1
        assert sum(owner.size for owner in analyze._audio_owners.values()) == 1024
        assert not tiny_audio.paths[-1].exists()
        assert (await client.get(url)).status_code == 200
        assert not analyze._audio_owners
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).content == b"x"
        assert not analyze._audio_owners


@pytest.mark.asyncio
async def test_concurrent_admissions_cannot_overbook(tiny_audio):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        results = await asyncio.gather(*(offer(client) for _ in range(4)))
        accepted = [event for event in results if event["stage"] == "groq_fallback"]
        refused = [event for event in results if event["stage"] != "groq_fallback"]
        assert len(accepted) == len(refused) == 2
        for event in refused:
            assert_busy(event)
        assert sum(p.exists() for p in tiny_audio.paths) == 2
        assert len(analyze._audio_owners) == 2
        assert sum(owner.size for owner in analyze._audio_owners.values()) == 2048
        for event in accepted:
            assert (await client.get(offered_url(event))).content == tiny_audio.data
        assert not analyze._audio_owners
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).status_code == 200


@pytest.mark.asyncio
async def test_expiry_frees_budget_once_and_new_offer_survives(tiny_audio, monkeypatch):
    expired = asyncio.Event()
    original = analyze._expire_temp_audio

    def expire(audio_id):
        original(audio_id)
        expired.set()

    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_TTL", 0.001)
    monkeypatch.setattr(analyze, "_expire_temp_audio", expire)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        await asyncio.wait_for(expired.wait(), timeout=1)
        assert not analyze._audio_owners
        assert not tiny_audio.paths[0].exists()
        assert (await client.get(url)).status_code == 404
        monkeypatch.setattr(analyze, "_TEMP_AUDIO_TTL", 300)
        reused = offered_url(await offer(client))
        original(url.rsplit("/", 1)[-1])
        analyze._release_temp_audio(url.rsplit("/", 1)[-1])
        assert len(analyze._audio_owners) == 1
        assert (await client.get(reused)).content == tiny_audio.data
        assert not analyze._audio_owners


@pytest.mark.asyncio
async def test_expiry_does_not_free_active_budget(tiny_audio, monkeypatch):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        async with hold_download(url):
            analyze._expire_temp_audio(url.rsplit("/", 1)[-1])
            assert len(analyze._audio_owners) == 1
            assert_busy(await offer(client))
            assert tiny_audio.paths[0].exists()
            assert not tiny_audio.paths[-1].exists()
        assert not analyze._audio_owners
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, OSError, asyncio.CancelledError])
async def test_send_failure_disconnect_cancel_reuse_budget(
    tiny_audio, monkeypatch, failure
):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))

        async def send(message):
            if message["type"] == "http.response.body":
                raise failure("synthetic failed/disconnected send")

        with pytest.raises(failure):
            await app(audio_scope(url.rsplit("/", 1)[-1]), receive_request, send)
        assert not analyze._audio_owners
        assert not tiny_audio.paths[0].exists()
        assert (await client.get(url)).status_code == 404
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).content == tiny_audio.data
        assert not analyze._audio_owners


@pytest.mark.asyncio
async def test_cancel_held_send_reuses_budget(tiny_audio, monkeypatch):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    entered = asyncio.Event()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))

        async def send(message):
            if message["type"] == "http.response.body":
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(
            app(audio_scope(url.rsplit("/", 1)[-1]), receive_request, send)
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert len(analyze._audio_owners) == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not analyze._audio_owners
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "range_header,status", [("invalid", 400), ("bytes=9999-", 416)]
)
async def test_range_error_frees_capacity(
    tiny_audio, monkeypatch, range_header, status
):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        assert (
            await client.get(url, headers={"range": range_header})
        ).status_code == status
        assert not analyze._audio_owners
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).status_code == 200


@pytest.mark.asyncio
async def test_missing_audio_releases_bytes_and_count(tiny_audio, monkeypatch):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        tiny_audio.paths[0].unlink()
        assert (await client.get(url)).status_code == 404
        assert not analyze._audio_owners
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).status_code == 200


@pytest.mark.asyncio
async def test_failed_unlink_keeps_charge_until_retry(tiny_audio, monkeypatch):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    original = analyze._remove_audio_file
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        path = tiny_audio.paths[0]

        def remove(other):
            return False if other == path else original(other)

        monkeypatch.setattr(analyze, "_remove_audio_file", remove)
        assert (await client.get(url)).status_code == 200
        owner = next(iter(analyze._audio_owners.values()))
        assert owner.size == 1024 and owner.retry is not None
        assert_busy(await offer(client))
        assert not tiny_audio.paths[-1].exists()
        monkeypatch.setattr(analyze, "_remove_audio_file", original)
        audio_id = url.rsplit("/", 1)[-1]
        analyze._release_temp_audio(audio_id)
        analyze._release_temp_audio(audio_id)
        assert owner.retry.cancelled()
        assert not analyze._audio_owners and not path.exists()
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).status_code == 200


@pytest.mark.asyncio
async def test_parent_directory_failure_does_not_hold_file_budget(
    tiny_audio, monkeypatch
):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        sibling = tiny_audio.paths[0].parent / "other-file.txt"
        sibling.write_bytes(b"x")
        assert (await client.get(url)).status_code == 200
        assert not tiny_audio.paths[0].exists() and sibling.exists()
        assert not analyze._audio_owners
        reused = offered_url(await offer(client))
        assert (await client.get(reused)).status_code == 200


@pytest.mark.asyncio
async def test_refused_file_removed_before_busy_yield(tiny_audio, monkeypatch):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        accepted = offered_url(await offer(client))
        response = await analyze.analyze_video_stream(AnalyzeRequest(url=URL))
        iterator = response.body_iterator
        try:
            async for payload in iterator:
                event = json.loads(payload)
                if event["stage"] == "error":
                    assert_busy(event)
                    break
            else:
                pytest.fail("Expected busy event")
            assert len(tiny_audio.paths) == 2
            assert tiny_audio.paths[0].exists()
            assert not tiny_audio.paths[1].exists()
            assert not tiny_audio.paths[1].parent.exists()
            assert len(analyze._audio_owners) == 1
        finally:
            await iterator.aclose()
        assert (await client.get(accepted)).status_code == 200


@pytest.mark.asyncio
async def test_failed_unlink_retry_runs_without_new_request(tiny_audio, monkeypatch):
    original = analyze._remove_audio_file
    recovered = asyncio.Event()
    failing = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        url = offered_url(await offer(client))
        path = tiny_audio.paths[0]

        def remove(other):
            if other == path and failing:
                return False
            removed = original(other)
            if other == path and removed:
                recovered.set()
            return removed

        monkeypatch.setattr(analyze, "_remove_audio_file", remove)
        assert (await client.get(url)).status_code == 200
        assert len(analyze._audio_owners) == 1
        assert path.exists()
        failing = False
        await asyncio.wait_for(recovered.wait(), timeout=3)
        assert not path.exists()
        assert not analyze._audio_owners


@pytest.mark.asyncio
async def test_refused_unlink_failure_is_best_effort_not_accepted_ownership(
    tiny_audio, monkeypatch
):
    monkeypatch.setattr(analyze, "_TEMP_AUDIO_MAX_FILES", 1)
    original = analyze._remove_audio_file
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        accepted = offered_url(await offer(client))
        accepted_path = tiny_audio.paths[0]

        def remove(path):
            return original(path) if path == accepted_path else False

        monkeypatch.setattr(analyze, "_remove_audio_file", remove)
        assert_busy(await offer(client))
        assert len(analyze._audio_owners) == 1
        assert tiny_audio.paths[1].exists()  # Explicit filesystem-failure limitation.
        monkeypatch.setattr(analyze, "_remove_audio_file", original)
        assert original(tiny_audio.paths[1])
        assert (await client.get(accepted)).status_code == 200
        assert not analyze._audio_owners
