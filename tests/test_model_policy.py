"""Model selection policy exercised through real analysis boundaries."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from podmemory.services import analyzer, model_catalog
from podmemory.services.transcript import from_user_paste


@pytest.fixture
def catalog_entries():
    return [
        {
            "id": "catalog/free-text",
            "name": "Free text",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        },
        {
            "id": "catalog/paid",
            "pricing": {"prompt": "0.01", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        },
        {
            "id": "catalog/image",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["image"]},
        },
    ]


@pytest.fixture
def provider(monkeypatch, catalog_entries):
    monkeypatch.setattr(analyzer.settings, "openrouter_api_key", "synthetic-test-only")
    monkeypatch.setattr(analyzer.settings, "llm_model", "operator/default")
    analyzer._analysis_cache.clear()
    monkeypatch.setattr(model_catalog, "_models_cache", None)
    state = {
        "calls": [],
        "fail_models": set(),
        "catalog_calls": 0,
        "fail_catalog": False,
        "clock": 10.0,
        "catalog_gate": None,
        "catalog_entered": asyncio.Event(),
    }
    monkeypatch.setattr(
        model_catalog, "time", SimpleNamespace(monotonic=lambda: state["clock"])
    )
    monkeypatch.setattr(model_catalog, "_models_lock", asyncio.Lock())

    async def send(client, request, **kwargs):
        if request.url.path == "/api/v1/models":
            state["catalog_calls"] += 1
            state["catalog_entered"].set()
            if state["catalog_gate"] is not None:
                await state["catalog_gate"].wait()
            if state["fail_catalog"]:
                return httpx.Response(503, request=request)
            return httpx.Response(
                200,
                request=request,
                json={"data": catalog_entries},
            )
        assert request.url.path == "/api/v1/chat/completions"
        model = json.loads(request.content)["model"]
        state["calls"].append(model)
        if model in state["fail_models"]:
            return httpx.Response(
                503, request=request, json={"error": {"message": "Unavailable"}}
            )
        result = {
            "title": "Study",
            "tldr": "Summary",
            "key_insights": ["Idea"],
            "flashcards": [{"q": "Question?", "a": "Answer."}],
        }
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": json.dumps(result)}}],
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["unknown/paid", "unknown/model:free"])
async def test_unknown_model_cannot_use_server_provider(provider, model):
    with pytest.raises(analyzer.AnalysisError, match="Model is not available"):
        await analyzer.analyze_transcript(from_user_paste("Study material."), model)
    assert provider["calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["", "operator/default", "catalog/free-text"])
async def test_operator_default_and_published_free_choice_work(provider, model):
    result = await analyzer.analyze_transcript(
        from_user_paste("Study material."), model
    )
    assert result["title"] == "Study"
    assert provider["calls"] == [model or "operator/default"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["catalog/paid", "catalog/image"])
async def test_nonfree_or_nontext_catalog_entry_is_rejected(provider, model):
    with pytest.raises(analyzer.AnalysisError, match="Model is not available"):
        await analyzer.analyze_transcript(from_user_paste("Study material."), model)
    assert provider["calls"] == []


@pytest.mark.asyncio
async def test_expired_catalog_failure_cannot_authorize_cached_analysis(provider):
    text = from_user_paste("Study material.")
    await analyzer.analyze_transcript(text, "catalog/free-text")
    provider["clock"] += model_catalog.CACHE_TTL + 1
    provider["fail_catalog"] = True
    with pytest.raises(analyzer.AnalysisError, match="Model is not available"):
        await analyzer.analyze_transcript(text, "catalog/free-text")
    assert provider["calls"] == ["catalog/free-text"]
    assert provider["catalog_calls"] == 2


@pytest.mark.asyncio
async def test_catalog_failure_does_not_disable_operator_default(provider):
    provider["fail_catalog"] = True
    result = await analyzer.analyze_transcript(from_user_paste("Study material."))
    assert result["model_used"] == "operator/default"
    assert provider["catalog_calls"] == 0


@pytest.mark.parametrize(
    "path", ["/api/analyze", "/api/analyze-chapter", "/api/analyze-stream"]
)
def test_api_rejects_unknown_model_without_provider_call(client, provider, path):
    response = client.post(
        path,
        json={
            "url": "",
            "text": "Study material.",
            "model": "unknown/paid",
            "source_type": "book",
        },
    )
    if path.endswith("-stream"):
        assert response.status_code == 200
        assert '"stage": "error"' in response.text
        assert '"stage": "done"' not in response.text
    else:
        assert response.status_code == 502
        assert response.json()["detail"] == "Analysis failed. Try a different model."
    assert provider["calls"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_config_catalog_and_analysis_share_one_fresh_lookup(provider, failed):
    provider["fail_catalog"] = failed
    provider["catalog_gate"] = asyncio.Event()
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(model_catalog.fetch_free_models()) for _ in range(3)]
        await provider["catalog_entered"].wait()
        provider["catalog_gate"].set()
    assert all(
        [entry["id"] for entry in task.result()]
        == ([] if failed else ["catalog/free-text"])
        for task in tasks
    )
    await analyzer.analyze_transcript(
        from_user_paste("Study material."),
        "operator/default" if failed else "catalog/free-text",
    )
    assert provider["catalog_calls"] == 1
    if failed:
        provider["fail_catalog"] = False
        provider["clock"] += 4
        with pytest.raises(analyzer.AnalysisError, match="Model is not available"):
            await analyzer.analyze_transcript(
                from_user_paste("Study material."), "catalog/free-text"
            )
        assert provider["catalog_calls"] == 1
        provider["clock"] += 1
        result = await analyzer.analyze_transcript(
            from_user_paste("Study material."), "catalog/free-text"
        )
        assert result["model_used"] == "catalog/free-text"
        assert provider["catalog_calls"] == 2


@pytest.mark.asyncio
async def test_catalog_refresh_revokes_a_previously_free_cached_model(
    provider, catalog_entries
):
    transcript = from_user_paste("Study material.")
    await analyzer.analyze_transcript(transcript, "catalog/free-text")
    catalog_entries[0]["pricing"]["completion"] = "0.01"
    provider["clock"] += model_catalog.CACHE_TTL + 1
    with pytest.raises(analyzer.AnalysisError, match="Model is not available"):
        await analyzer.analyze_transcript(transcript, "catalog/free-text")
    assert provider["calls"] == ["catalog/free-text"]


def test_config_only_publishes_accepted_free_text_models(client, provider):
    config = client.get("/api/config").json()
    assert [entry["id"] for entry in config["models"]] == ["catalog/free-text"]
    response = client.post(
        "/api/analyze-chapter",
        json={
            "text": "Study material.",
            "model": config["default_model"],
            "source_type": "book",
        },
    )
    assert response.status_code == 200
    assert response.json()["model_used"] == "catalog/free-text"
    assert provider["catalog_calls"] == 1


@pytest.mark.asyncio
async def test_unlisted_fallback_cannot_use_server_provider(provider):
    provider["fail_models"].add("catalog/free-text")
    with pytest.raises(analyzer.AnalysisError):
        await analyzer.analyze_transcript(
            from_user_paste("Study material."), "catalog/free-text"
        )
    assert provider["calls"] == ["catalog/free-text"]


@pytest.mark.asyncio
async def test_free_fallback_succeeds_without_trying_unlisted_predecessor(
    provider, catalog_entries
):
    provider["fail_models"].add("catalog/free-text")
    catalog_entries.append(
        {
            "id": "meta-llama/llama-3.3-8b-instruct:free",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        }
    )
    result = await analyzer.analyze_transcript(
        from_user_paste("Study material."), "catalog/free-text"
    )
    assert result["model_used"] == "meta-llama/llama-3.3-8b-instruct:free"
    assert provider["calls"] == [
        "catalog/free-text",
        "meta-llama/llama-3.3-8b-instruct:free",
    ]
    assert provider["catalog_calls"] == 1


@pytest.mark.asyncio
async def test_stale_catalog_failure_never_authorizes_fallback(
    provider, catalog_entries
):
    catalog_entries.append(
        {
            "id": "google/gemini-2.0-flash-001",
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"output_modalities": ["text"]},
        }
    )
    await model_catalog.fetch_free_models()
    provider["clock"] += model_catalog.CACHE_TTL + 1
    provider["fail_catalog"] = True
    provider["fail_models"].add("operator/default")
    with pytest.raises(analyzer.AnalysisError):
        await analyzer.analyze_transcript(from_user_paste("Study material."))
    assert provider["calls"] == ["operator/default"]
    assert provider["catalog_calls"] == 2


@pytest.mark.asyncio
async def test_explicit_operator_default_remains_allowed_as_fallback(
    provider, monkeypatch
):
    monkeypatch.setattr(analyzer.settings, "llm_model", "google/gemini-2.0-flash-001")
    provider["fail_models"].add("catalog/free-text")
    result = await analyzer.analyze_transcript(
        from_user_paste("Study material."), "catalog/free-text"
    )
    assert result["model_used"] == "google/gemini-2.0-flash-001"
    assert provider["calls"] == ["catalog/free-text", "google/gemini-2.0-flash-001"]
