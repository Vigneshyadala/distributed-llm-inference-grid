# Engine loop and HTTP gateway

Two pieces connect clients to the scheduler: `src/grid/engine.py` runs the steps, and
`src/grid/gateway.py` is the FastAPI front door.

```
client --HTTP--> gateway --submit--> Engine --next_step--> Scheduler
                                       |                      ^
                                       +-- execute(plan) -----+-- complete_step(tokens)
                                       |
                                       +-- events --> the client's Job (token, token, ..., done)
```

## The engine loop

- **One loop, many clients.** `Engine.submit(prompt_len, max_new_tokens)` returns a `Job` at once.
  The loop asks the scheduler for the next step, runs it, reports the tokens, and gives each client
  its token as an event. A client reads its job with `async for event in job.events()` or waits for
  the whole reply with `await job.result()`.
- **The model never blocks the loop.** `execute(plan)` is an ordinary blocking function and runs in
  a worker thread, so new requests are accepted while a step is running.
- **Bad steps do not kill the server.** If the model raises, only the requests in that step get an
  `error` event and their blocks are freed (`Scheduler.fail_step`); everyone else carries on.
- **Back-pressure.** At most `max_waiting` requests may wait. After that `submit` raises
  `QueueFull`, which the gateway turns into HTTP 503 with `Retry-After`.
- **Cancel.** `cancel(job)` (the client went away) aborts the request before the next step and
  frees its blocks.

## The stand-in model

The real tensor-parallel model is connected to the master later. Until then `SimulatedModel` stands
in: every step stores keys and values for the planned tokens in the real paged cache and answers
the dummy token `1000 + position`. Scheduling, batching, memory use, block reuse and refusals are
therefore real; only the token values are not. `--step-seconds` makes a step take time so batching
can be watched.

## Endpoints

| Method | Path | Answer |
|--------|------|--------|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/stats` | scheduler counts, cache memory accounting and the last 20 steps (what the dashboard will read) |
| POST | `/v1/generate` | body `{"prompt": "...", "max_new_tokens": 32}`; the generated tokens, the finish reason and wait and latency in steps |

| Status | Meaning |
|--------|---------|
| 200 | the reply |
| 413 | the request could never fit in the cache, even when it is empty |
| 422 | bad body (empty prompt, `max_new_tokens` outside 1 to 512, wrong types) |
| 503 | too many requests waiting; retry after a second |
| 500 | the model failed during this request |

## Try it

```
python scripts/serve.py
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/v1/generate -H "Content-Type: application/json" -d "{\"prompt\": \"hello grid\", \"max_new_tokens\": 5}"
curl http://127.0.0.1:8765/stats
```

Send several requests at once and watch `/stats`: `recent_steps` shows them running in the same
batch.

## Limits

- The tokenizer is a stand-in that counts words. The model's real tokenizer replaces it with the real model.
- A reply is returned when it is complete. Streaming tokens as they appear, over a WebSocket, is next.
- Steps are counted, not timed; real timings come with the load tests.
- The worker thread and the event loop both touch the cache. The loop only adds and removes
  sequences between steps, and a step only writes into sequences that already exist, so they never
  change the same structure at the same time. `/stats` reads are for display only.
