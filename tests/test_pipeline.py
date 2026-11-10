"""End-to-end integration tests for the RAG pipeline and FastAPI endpoints."""

import pytest
import httpx
from httpx import ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["total_documents"] == 3


@pytest.mark.asyncio
async def test_metrics_endpoint():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "p50_latency_ms" in data


@pytest.mark.asyncio
async def test_ask_factual_question():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "question": "What is the dimensionality of d_model in the base Transformer?",
            "retrieval_setting": "B",
            "stream": False,
        }
        resp = await client.post("/ask", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["refusal"] is False
        assert "512" in data["answer"]
        assert len(data["citations"]) > 0
        assert any("Attention Is All You Need" in c["paper_title"] for c in data["citations"])
        assert len(data["pipeline_stages"]) >= 4
        assert len(data["tool_calls"]) >= 1


@pytest.mark.asyncio
async def test_ask_out_of_scope_refusal():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "question": "What is the battery life of the Apple Vision Pro?",
            "retrieval_setting": "B",
            "stream": False,
        }
        resp = await client.post("/ask", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["refusal"] is True
        assert "REFUSAL:" in data["answer"]


@pytest.mark.asyncio
async def test_ask_streaming():
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "question": "What is d_model in the Transformer?",
            "retrieval_setting": "B",
            "stream": True,
        }
        resp = await client.post("/ask", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert "data:" in text
        assert "[DONE]" in text

