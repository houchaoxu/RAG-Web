"""Integration tests for API endpoints."""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from raganything.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_endpoint(client):
    res = await client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_get_config_returns_defaults(client):
    res = await client.get("/api/config/llm")
    assert res.status_code == 200
    data = res.json()
    assert "provider" in data


@pytest.mark.asyncio
async def test_get_full_config(client):
    res = await client.get("/api/config/full")
    assert res.status_code == 200
    data = res.json()
    assert "llm" in data
    assert "embedding" in data
    assert "rag" in data


@pytest.mark.asyncio
async def test_list_providers(client):
    res = await client.get("/api/config/llm/providers")
    assert res.status_code == 200
    data = res.json()
    assert "openai" in data["providers"]


@pytest.mark.asyncio
async def test_frontend_served(client):
    res = await client.get("/")
    assert res.status_code == 200
    assert "RAG-Anything" in res.text


@pytest.mark.asyncio
async def test_query_without_init_returns_error(client):
    res = await client.post("/api/query", json={"query": "test", "mode": "mix"})
    assert res.status_code == 500


@pytest.mark.asyncio
async def test_process_without_init_returns_error(client):
    res = await client.post("/api/documents/process", json={"file_path": "/nonexistent"})
    assert res.status_code == 500
