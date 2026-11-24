"""The HTTP gateway: how clients reach the engine.

    GET  /health         is the server up
    GET  /stats          scheduler, cache and the last steps (what the dashboard will read)
    POST /v1/generate    {"prompt": "...", "max_new_tokens": 32} -> the generated tokens
    WS   /v1/stream      send the same JSON once; every token arrives as soon as it exists
    WS   /ws/events      a live feed of everything the engine does (the dashboard's source)

On the streaming socket the same problems arrive as a last message {"type": "error", "status": 422 | 503,
"detail": ...} (or {"type": "rejected", ...}) and the server closes the socket. Closing the socket
while a reply is running cancels the request and frees its blocks.

Errors are plain HTTP statuses: 422 for a bad request body, 413 for a request that can never fit
in the cache, 503 (with Retry-After) when too many requests are already waiting, 500 if the model
failed during the request.

The tokenizer is a stand-in that counts words; it is replaced by the model's real tokenizer when
the real model is connected.
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

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


async def _wait_for_disconnect(ws):
    """Returns when the client closes the socket (anything it sends is ignored)."""
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):
        return


async def _send_until_done(ws, next_event):
    """Send events to the client until the stream ends (next_event gives None) or the client leaves.

    Returns True if the stream ended and False if the client left first.
    """
    watcher = asyncio.create_task(_wait_for_disconnect(ws))
    try:
        while True:
            getter = asyncio.ensure_future(next_event())
            done, _ = await asyncio.wait({getter, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                getter.cancel()
                return False
            event = getter.result()
            if event is None:
                return True
            await ws.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        return False
    finally:
        watcher.cancel()


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

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            body = GenerateBody.model_validate(await ws.receive_json())
        except WebSocketDisconnect:
            return
        except (ValidationError, ValueError) as exc:
            detail = exc.errors(include_url=False, include_context=False, include_input=False) \
                if isinstance(exc, ValidationError) else "the first message must be a JSON object"
            await ws.send_json({"type": "error", "status": 422, "detail": detail})
            await ws.close(code=1008)
            return
        try:
            job = engine.submit(max(1, len(tokenizer(body.prompt))), body.max_new_tokens)
        except QueueFull as exc:
            await ws.send_json({"type": "error", "status": 503, "detail": str(exc)})
            await ws.close(code=1013)
            return
        events = job.events()

        async def next_event():
            return await anext(events, None)

        finished = await _send_until_done(ws, next_event)
        if not finished:
            engine.cancel(job)               # the client left: free its place in the batch
        else:
            try:
                await ws.close()
            except RuntimeError:             # the client closed at the same moment
                pass

    @app.websocket("/ws/events")
    async def live_events(ws: WebSocket):
        await ws.accept()
        queue = engine.subscribe()
        try:
            scheduler = engine.scheduler
            await ws.send_json({"type": "snapshot", "stats": scheduler.stats(),
                                "cache": scheduler.cache.stats(), "recent_steps": scheduler.timeline[-20:]})
            await _send_until_done(ws, queue.get)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            engine.unsubscribe(queue)

    return app
