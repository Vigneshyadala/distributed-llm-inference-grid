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
| WS | `/v1/stream` | send the same JSON once; every token arrives as soon as it exists, then `done` |
| WS | `/ws/events` | a live feed of everything the engine does: the source of the dashboard |
| GET | `/` | the live dashboard page ([dashboard.md](dashboard.md)); its files are under `/assets/` |

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

## Streaming one reply

`/v1/stream` is a WebSocket. The client sends one JSON message, `{"prompt": "...", "max_new_tokens": 8}`,
and first receives the request's id, then one message per token the moment it exists, so the first
token shows up after the first step instead of after the whole reply:

```
{"type": "accepted", "id": "req-1", "prompt_tokens": 3}
{"type": "token", "token": 1000, "index": 0}
{"type": "token", "token": 1001, "index": 1}
...
{"type": "done", "reason": "length"}
```

Problems arrive as a last message and the server closes the socket: `{"type": "error", "status": 422,
"detail": ...}` for a bad request, `{"type": "error", "status": 503, ...}` when the queue is full,
`{"type": "rejected", "reason": ...}` when the request can never fit. **If the client closes the
socket while a reply is running, the request is cancelled and its blocks are freed at once.**

## The live event feed

`/ws/events` is what the dashboard listens to. It starts with one `snapshot` (the same numbers as
`/stats`) and then sends every event as it happens, each with a timestamp `t`:

| Event | When | Main fields |
|-------|------|-------------|
| `submitted` | a request is queued | `id`, `prompt_len`, `max_new_tokens` |
| `rejected` | it can never fit in the cache | `id`, `reason` |
| `step` | after every step | `step`, `prefill`, `decode`, `finished`, `running`, `waiting`, `reserved_blocks`, `blocks_used`, `tokens_in`, `ms`, `stats`, `block_map` (who owns each cache block after the step) |
| `finished` | a request ends | `id`, `kind` (`done`, `cancelled`, `error`), `reason`, `tokens`, `wait_steps`, `latency_steps` |
| `stopped` | the engine shuts down | |

A watcher that reads too slowly loses its **oldest** events (each has a bounded queue), so a slow
browser tab can never slow the engine down. Any number of watchers can connect at once.

## Watching from a terminal

```
python scripts/serve.py --step-seconds 0.05
python scripts/stream_client.py --events                     follow every step (second window)
python scripts/stream_client.py "hello grid" --max-new 20    stream one reply (third window)
```

Start two replies together and the event window shows the second one joining the same batch while
the first is still decoding:

```
step    0  prefill [req-1]  decode [-]  running 1  waiting 1  blocks 1/256  51.4 ms
step    1  prefill [req-2]  decode [req-1]  running 2  waiting 0  blocks 2/256  50.9 ms
step    2  prefill [-]  decode [req-1,req-2]  running 2  waiting 0  blocks 2/256  50.8 ms
```

## Limits

- The tokenizer is a stand-in that counts words. The model's real tokenizer replaces it with the real model.
- The `ms` of a step is the simulated model's time. Real step times come with the real model.
- The event feed shows what happens from the moment you connect (plus the last 20 steps in the
  snapshot); it is not a stored history.
- Steps are counted, not timed; real timings come with the load tests.
- The worker thread and the event loop both touch the cache. The loop only adds and removes
  sequences between steps, and a step only writes into sequences that already exist, so they never
  change the same structure at the same time. `/stats` reads are for display only.
