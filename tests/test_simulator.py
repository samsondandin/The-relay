"""Unit tests for the ModelSimulator."""

import pytest
from app.simulator import ModelSimulator


@pytest.mark.asyncio
async def test_simulator_deterministic_seeding():
    sim1 = ModelSimulator(random_seed=42, base_latency_ms=10.0, latency_jitter_ms=5.0)
    sim2 = ModelSimulator(random_seed=42, base_latency_ms=10.0, latency_jitter_ms=5.0)

    resp1 = await sim1.generate("What is d_model?")
    resp2 = await sim2.generate("What is d_model?")

    assert resp1.content == resp2.content
    assert resp1.tokens_in == resp2.tokens_in
    assert resp1.tokens_out == resp2.tokens_out


@pytest.mark.asyncio
async def test_simulator_failure_injection():
    sim = ModelSimulator(failure_probability=1.0, base_latency_ms=10.0)

    with pytest.raises(ConnectionError) as exc_info:
        await sim.generate("Any prompt")

    assert "Simulated upstream 503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_simulator_streaming():
    sim = ModelSimulator(base_latency_ms=10.0, streaming_chunk_delay_ms=2.0)
    chunks = []
    async for chunk in sim.stream_generate("What is d_model in the Transformer?"):
        chunks.append(chunk["delta"])

    assert len(chunks) > 5
    assert "d_model = 512" in "".join(chunks)


@pytest.mark.asyncio
async def test_simulator_refusal_generation():
    sim = ModelSimulator(base_latency_ms=10.0)
    resp = await sim.generate("What is the battery life of Apple Vision Pro?")
    assert resp.refusal is True
    assert "REFUSAL:" in resp.content
