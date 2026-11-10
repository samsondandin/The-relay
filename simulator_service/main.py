"""Standalone Simulator Microservice reproducing Groq API behavior locally."""

import time
import random
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Standalone Groq/LLM Stand-in Service", version="1.0.0")


class SimConfig(BaseModel):
    base_latency_ms: float = 280.0
    latency_jitter_ms: float = 50.0
    failure_probability: float = 0.0
    timeout_probability: float = 0.0
    rpm_limit: int = 60
    tpm_limit: int = 15000
    daily_request_budget: int = 1000
    load_slowdown_factor: float = 0.04
    streaming_chunk_delay_ms: float = 10.0


cfg = SimConfig()
in_flight = 0
request_times = []


@app.get("/health")
def health():
    return {"status": "ok", "service": "simulator", "in_flight": in_flight}


@app.get("/config")
def get_config():
    return cfg.dict()


@app.post("/config")
def update_config(new_cfg: SimConfig):
    global cfg
    cfg = new_cfg
    return {"status": "updated", "config": cfg.dict()}


@app.post("/v1/chat/completions")
async def chat_completions(payload: Dict[str, Any]):
    global in_flight
    in_flight += 1

    try:
        if cfg.failure_probability > 0 and random.random() < cfg.failure_probability:
            raise HTTPException(status_code=503, detail="Simulated service unavailable")

        # Simulate dynamic slowdown based on concurrent in-flight requests
        slowdown = 1.0 + (in_flight * cfg.load_slowdown_factor)
        jitter = random.uniform(-cfg.latency_jitter_ms, cfg.latency_jitter_ms)
        target_ms = max(50.0, (cfg.base_latency_ms + jitter) * slowdown)

        messages = payload.get("messages", [])
        user_prompt = messages[-1]["content"] if messages else ""

        # Simulated response synthesis
        ans = (
            f"Simulated technical completion citing Attention Is All You Need (arXiv:1706.03762v7, p. 3) "
            f"and Dynamic Chunking (arXiv:2507.07955v2, p. 4) for query context."
        )

        stream = payload.get("stream", False)
        if stream:
            async def event_generator():
                words = ans.split()
                # Initial TTFT delay
                await asyncio.sleep((target_ms * 0.35) / 1000.0)
                for w in words:
                    yield f"data: {{\"choices\": [{{\"delta\": {{\"content\": \"{w} \"}}}}]}}\n\n"
                    await asyncio.sleep(cfg.streaming_chunk_delay_ms / 1000.0)
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        await asyncio.sleep(target_ms / 1000.0)
        return {
            "id": f"chatcmpl-sim-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "simulated-groq"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ans},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_prompt.split()) + 20,
                "completion_tokens": len(ans.split()),
                "total_tokens": len(user_prompt.split()) + len(ans.split()) + 20,
            },
        }

    finally:
        in_flight = max(0, in_flight - 1)

