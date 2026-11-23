"""The HTTP gateway: how clients reach the engine.

    GET  /health         is the server up
    GET  /stats          scheduler, cache and the last steps (what the dashboard will read)
    POST /v1/generate    {"prompt": "...", "max_new_tokens": 32} -> the generated tokens

Errors are plain HTTP statuses: 422 for a bad request body, 413 for a request that can never fit
in the cache, 503 (with Retry-After) when too many requests are already waiting, 500 if the model
failed during the request.

The tokenizer is a stand-in that counts words; it is replaced by the model's real tokenizer when
the real model is connected. Streaming tokens to the browser comes next, over a WebSocket.
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from grid.engine import Engine, QueueFull, SimulatedModel
from grid.kvcache import PagedKVCache
from grid.scheduler import Scheduler


def word_tokenizer(text):
    """Stand-in tokenizer: one token per word."""
    return text.split()


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=32, ge=1, le=512)


def build_engine(blocks=256, block_size=16, layers=2, kv_heads=2, head_dim=8,
                 max_batch_size=8, max_prefill_tokens=None, max_waiting=64, step_seconds=0.0):
    """An engine on the simulated model, with a real paged cache of the given size."""
    cache = PagedKVCache(layers, kv_heads, head_dim, block_size, blocks)
    scheduler = Scheduler(cache, max_batch_size=max_batch_size, max_prefill_tokens=max_prefill_tokens)
    return Engine(scheduler, SimulatedModel(cache, step_seconds), max_waiting=max_waiting)


def create_app(engine, tokenizer=word_tokenizer):
    @asynccontextmanager
    async def lifespan(app):
        await engine.start()
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(title="Distributed Memory Grid gateway", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        scheduler = engine.scheduler
        return {"scheduler": scheduler.stats(), "cache": scheduler.cache.stats(),
                "recent_steps": scheduler.timeline[-20:]}

    @app.post("/v1/generate")
    async def generate(body: GenerateBody):
        prompt_tokens = max(1, len(tokenizer(body.prompt)))
        try:
            job = engine.submit(prompt_tokens, body.max_new_tokens)
        except QueueFull as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503, headers={"Retry-After": "1"})
        try:
            result = await job.result()
        except asyncio.CancelledError:       # the client disconnected: free its place in the batch
            engine.cancel(job)
            raise
        request = job.request
        if result["status"] == "done":
            return {"id": request.id, "prompt_tokens": prompt_tokens, "tokens": result["tokens"],
                    "finish_reason": result["reason"],
                    "wait_steps": request.wait_steps, "latency_steps": request.latency_steps}
        status = {"rejected": 413, "error": 500}.get(result["status"], 503)
        return JSONResponse({"detail": result["reason"], "id": request.id}, status_code=status)

    return app
