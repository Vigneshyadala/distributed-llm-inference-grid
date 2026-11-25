# The live dashboard

Open `http://127.0.0.1:8765/` while `python scripts/serve.py` is running. The page is plain HTML,
CSS and JavaScript (`src/grid/web/`), served by the gateway itself, with no build step.

## What it shows

| Section | Source | What you see |
|---------|--------|--------------|
| Hero numbers | `/ws/events` | steps run, requests running, tokens per second (measured over the last 3 seconds), cache in use |
| Request path | `/ws/events` | the scheduler's running and waiting counts and the cache's block count; the dots speed up while there is work |
| Continuous batching | `step` events | one row per request, one column per step: **P** prefill, **d** decode, a dotted line while waiting for room, a check mark when done, a cross for a request that could never fit |
| Paged KV cache | `block_map` in the events | every cache block, coloured by the request that owns it, filled by how full it is, and flashing when it changes owner; plus gauges for blocks in use and blocks reserved |
| Try it | `/v1/stream` | send a prompt and watch the tokens arrive; "Send 4 requests" fires four at once so the lanes and the block map fill up |
| Live terminal | `/ws/events` | the engine's events as text, like `tail -f` |

A request keeps the same colour everywhere: in its lane, on its blocks and on its token card.

## Live, recorded and offline

The badge in the top bar says where the numbers come from.

- **LIVE** (crimson): connected to a running engine. Everything is what it is doing now.
- **RECORDED RUN** (amber): no server answered, or you pressed "Replay recorded run". The page replays
  `assets/recorded.json`, a real run of the engine saved to a file, and loops it. Sending requests is
  disabled because nothing is listening. The page keeps trying to reconnect and switches back to
  live by itself.
- **OFFLINE** (grey): the connection dropped; the page is retrying.

Because the replay needs no server, the same page can be hosted as plain static files (for example on
GitHub Pages) and still show the system working, clearly marked as a recording.

## The recorded run

`python scripts/record_run.py` runs the real engine on the simulated model and saves what happened:
six requests arrive one after another at a 12-block cache (48 tokens), so one request has to wait
for room and one is refused because it could never fit. Run it again to record a new one. Only the
timing offsets are kept; the file is a few kilobytes.

## How it works

1. The page opens a WebSocket to `/ws/events`. The first message is a snapshot (counts, cache
   numbers, the last 20 steps and the block map); then every event arrives as it happens.
2. Each `step` event lists who prefilled and decoded, so the lanes are drawn from it directly, and
   carries the block map, so the memory view is drawn from it directly.
3. Opening the page in the middle of a run is fine: requests whose "finished" event was missed are
   inferred from the snapshot.
4. `prefers-reduced-motion` turns the animations off.

## Limits

- The workers in the picture are simulated until the real model is connected; the scheduler, the
  batching and the cache are real.
- The tokens are placeholders (1000, 1001, ...) from the simulated model.
- Planned next, once the real model sits behind the gateway: layer waterfall, all-reduce pulse with
  real timings, straggler glow, and a one-worker against two-worker comparison. They are listed as
  "coming next" on the page and are not drawn with made-up numbers.
