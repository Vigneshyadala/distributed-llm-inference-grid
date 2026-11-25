# Mid Review checklist

What is built, how to show it, and what to say about what is not built yet. Every command runs
from the repository root with the virtual environment active.

## Plan items

| Item | Status | Evidence |
|------|--------|----------|
| Master-worker topology and communication | done | `python scripts/hello_grid.py`, [grpc-protocol.md](grpc-protocol.md) |
| Weights loaded and sharded without loss | done | `python scripts/inspect_weights.py`, [weights.md](weights.md) |
| Tensor parallelism distributing weights correctly | done | `python scripts/tp_check.py`, `python scripts/block_check.py`, [tensor-parallel.md](tensor-parallel.md) |
| Paged KV cache | done | `python scripts/kv_demo.py --check`, [kv-cache.md](kv-cache.md) |
| Continuous batching | done (scheduler, on a simulated model) | `python scripts/batch_demo.py --check`, [batching.md](batching.md) |
| HTTP gateway, token streaming, live event feed | done | `python scripts/serve.py`, [gateway.md](gateway.md) |
| Live dashboard | first version | `http://127.0.0.1:8765/`, [dashboard.md](dashboard.md) |
| Distributed inference generating real tokens | next | the real model goes behind `execute(plan)` in `grid/engine.py` |

## A five-minute demo, in order

1. **Tests.** `python -m pytest -q` (all pass; takes about two minutes on the laptop).
2. **Tensor parallelism.** `python scripts/tp_check.py --workers 2` prints `TP CHECK OK`: two worker
   processes give the same answer as the full layers. `python scripts/block_check.py` does the same
   for a whole transformer block.
3. **Memory.** `python scripts/kv_demo.py` draws the block map as sequences grow, finish and free
   blocks, and prints the Qwen2.5-0.5B memory figures.
4. **Batching.** `python scripts/batch_demo.py --check` draws static against continuous batching for
   the same eight requests: 10 steps instead of 13.
5. **Everything together.** `python scripts/serve.py`, open `http://127.0.0.1:8765/`, press
   "Send 4 requests" and point at the lanes, the block map and the terminal. In a second window,
   `python scripts/stream_client.py --events` shows the same steps as text.

## Honest limits to state up front

- The model behind the gateway is a stand-in: it does real cache traffic and answers placeholder
  tokens. Connecting the real tensor-parallel model is the next step.
- The laptop has a 2-core CPU and no GPU, so tensor parallelism is shown for correctness and memory,
  not for speed. The backend is gloo; NCCL cannot run on this hardware.
- Timings in the dashboard are the simulated model's. Real timings come with the load tests.

## Commit days in this repository

| Date | Work |
|------|------|
| 17 and 18 Nov | Week 1: master-worker grid over gRPC, safetensors weight loading and sharding |
| 19 Nov | tensor-parallel layers |
| 20 Nov | tensor-parallel transformer block |
| 21 Nov | paged KV cache |
| 22 Nov | continuous batching scheduler |
| 23 Nov | engine loop and HTTP gateway |
| 24 Nov | live token streaming and event feed |
| 25 Nov | live dashboard and recorded run |
