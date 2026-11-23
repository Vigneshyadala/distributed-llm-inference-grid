"""The engine loop: runs the scheduler's steps one after another and hands tokens to clients.

    client --submit--> Engine --next_step--> Scheduler
                         |                      ^
                         +--execute(plan)-------+--complete_step(tokens)
                         |
                         +--events--> the client's Job (token, token, ..., done)

`execute(plan)` is the model: it takes a `StepPlan` and returns {request id: new token}. It is an
ordinary blocking function that runs in a worker thread, so a slow step never freezes the event
loop that accepts new requests. Today the model behind it is `SimulatedModel`, a stand-in that
stores keys and values in the real paged cache and answers dummy tokens; the master replaces it
with the real tensor-parallel model.

Every client owns a `Job` with a queue of events:

    {"type": "token", "token": 1004, "index": 0}        one per generated token
    {"type": "done", "reason": "length" | "eos"}        the last event of a good reply
    {"type": "rejected" | "error" | "cancelled", "reason": "..."}   the last event of a bad one
"""
import asyncio
import itertools
import time

import numpy as np

from grid.scheduler import Request

TERMINAL = ("done", "rejected", "error", "cancelled")


class QueueFull(RuntimeError):
    """Too many requests are already waiting; the client should retry shortly."""


class Job:
    """One client's view of one request: a queue of events."""

    def __init__(self, request):
        self.request = request
        self.queue = asyncio.Queue()
        self.finished = False

    def push(self, event):
        if not self.finished:
            self.queue.put_nowait(event)
            self.finished = event["type"] in TERMINAL

    async def events(self):
        """Yield events until (and including) the last one."""
        while True:
            event = await self.queue.get()
            yield event
            if event["type"] in TERMINAL:
                return

    async def result(self):
        """Wait for the whole reply: {"status", "tokens", "reason"}."""
        tokens = []
        async for event in self.events():
            if event["type"] == "token":
                tokens.append(event["token"])
            else:
                return {"status": event["type"], "tokens": tokens, "reason": event.get("reason")}


class Engine:
    """Runs a `Scheduler` with a model, serving many jobs at once."""

    def __init__(self, scheduler, execute, max_waiting=64):
        self.scheduler = scheduler
        self.execute = execute
        self.max_waiting = max_waiting
        self._jobs = {}
        self._cancelled = set()
        self._counter = itertools.count(1)
        self._wakeup = None
        self._task = None

    # ---------------------------------------------------------------- lifecycle

    async def start(self):
        if self._task is None:
            self._wakeup = asyncio.Event()
            self._task = asyncio.create_task(self._run(), name="engine-loop")

    async def stop(self):
        """Stop the loop; jobs still open are told they were cancelled."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for job in list(self._jobs.values()):
            job.push({"type": "cancelled", "reason": "engine stopped"})
        self._jobs.clear()

    # ------------------------------------------------------------------ clients

    def submit(self, prompt_len, max_new_tokens, request_id=None):
        """Queue a request and return its `Job`. Raises `QueueFull` or `ValueError`."""
        if self._task is None:
            raise RuntimeError("the engine is not running; call start() first")
        if len(self.scheduler.waiting) >= self.max_waiting:
            raise QueueFull(f"{self.max_waiting} requests are already waiting")
        request = Request(request_id if request_id is not None else f"req-{next(self._counter)}",
                          prompt_len, max_new_tokens)
        job = Job(request)
        self._jobs[request.id] = job
        self.scheduler.submit(request)
        if request.state == "rejected":
            job.push({"type": "rejected", "reason": request.reason})
            del self._jobs[request.id]
        else:
            self._wakeup.set()
        return job

    def cancel(self, job):
        """Stop serving a job (the client went away). Takes effect before the next step."""
        if job.request.id in self._jobs:
            self._cancelled.add(job.request.id)
            self._wakeup.set()

    # --------------------------------------------------------------------- loop

    async def _run(self):
        scheduler = self.scheduler
        while True:
            await self._wakeup.wait()
            while True:
                self._apply_cancellations()
                plan = scheduler.next_step()
                if plan is None:
                    break
                try:
                    tokens = await asyncio.to_thread(self.execute, plan)
                except asyncio.CancelledError:
                    scheduler.fail_step(plan, "engine stopped")
                    raise
                except Exception as exc:  # noqa: BLE001 - a bad step must not kill the engine
                    for request in scheduler.fail_step(plan, f"model error: {type(exc).__name__}: {exc}"):
                        self._finish(request, "error")
                    continue
                done = scheduler.complete_step(plan, tokens)
                for request in plan.requests:
                    job = self._jobs.get(request.id)
                    if job is not None:
                        job.push({"type": "token", "token": tokens[request.id], "index": request.generated - 1})
                for request in done:
                    self._finish(request, "done")
                await asyncio.sleep(0)       # let new requests and clients in between steps
            self._wakeup.clear()

    def _apply_cancellations(self):
        for request_id in list(self._cancelled):
            self._cancelled.discard(request_id)
            if self.scheduler.abort(request_id, "cancelled by the client"):
                self._finish(self.scheduler.get(request_id), "cancelled")

    def _finish(self, request, kind):
        job = self._jobs.pop(request.id, None)
        if job is not None:
            job.push({"type": kind, "reason": request.reason})


class SimulatedModel:
    """A stand-in for the real model: real cache traffic, dummy tokens.

    Each step stores keys and values for the planned tokens in the paged cache (so memory use,
    block reuse and `OutOfBlocks` are all real) and answers the token 1000 + its position.
    `step_seconds` makes a step take time, so streaming and batching can be watched.
    """

    def __init__(self, cache, step_seconds=0.0):
        self.cache = cache
        self.step_seconds = step_seconds

    def __call__(self, plan):
        if self.step_seconds:
            time.sleep(self.step_seconds)
        cache, tokens = self.cache, {}
        for request in plan.prefill + plan.decode:
            count = request.prompt_len if request in plan.prefill else 1
            data = np.zeros((1, cache.kv_heads, count, cache.head_dim), dtype=np.float32)
            for layer in range(cache.layers):
                cache.extend(request.id, layer, data, data)
            tokens[request.id] = 1000 + request.generated
        return tokens
