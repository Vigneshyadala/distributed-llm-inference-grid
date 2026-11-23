# Distributed Memory Grid for Real-Time LLM Inference

A custom distributed inference engine in Python. It shards a Large Language Model across several worker nodes with **tensor parallelism**, talks between nodes over **gRPC and asyncio**, manages a **KV cache** per worker and serves tokens to clients through a **FastAPI + WebSocket** gateway with **continuous batching**.

> Enterprise Project 1 of the Zaalima Development Pvt. Ltd. internship program (Month 2). Built by Vignesh Yadala.

## Status

| Week | Scope | State |
|------|-------|-------|
| 1 | Master-worker topology (asyncio + gRPC), safetensors weight loading with memory-mapping, commit rules | Done |
| 2 | Tensor parallelism with `torch.distributed`, asynchronous KV-cache manager, review cycles | Tensor-parallel layers and blocks done; paged KV cache done; review cycles next |
| 3 | Continuous batching, collective-communication backend switch, FastAPI + WebSocket gateway | Batching scheduler and HTTP gateway done; live streaming next |
| 4 | Docker Swarm deployment, load tests (TTFT, inter-node latency), final report | Planned |

## Design at a glance

```
  Client ──WebSocket──> Gateway (FastAPI) ──> Master (scheduler, continuous batching)
                                                  │  gRPC (asyncio)
                              ┌───────────────────┼───────────────────┐
                              ▼                   ▼                   ▼
                          Worker 0            Worker 1            Worker N
                        shard of weights    shard of weights    shard of weights
                        + its KV cache      + its KV cache      + its KV cache
                              └──── all-reduce (torch.distributed) ────┘
```

Details are in [docs/architecture.md](docs/architecture.md) and the dated plan is in [docs/plan.md](docs/plan.md).

## Setup

```
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
pip install --no-cache-dir -r requirements-dev.txt
python -m pytest -q
python scripts/check_env.py --allreduce
```

`check_env.py` changes nothing. It prints the Python, CPU, RAM and disk figures, the library versions, which PyTorch backends are available, and (with `--allreduce`) runs a two-process `gloo` all-reduce, the building block of tensor parallelism.

## Try the master-worker grid

```
python scripts/hello_grid.py --workers 2
```

This starts two separate worker processes on free local ports. The master greets all of them at once over gRPC (`grpc.aio` and `asyncio.gather`), checks that their ranks and world size are consistent, measures a ping round trip to each, and prints one row per worker with its own process id:

```
worker     rank world     pid version  address                 ping
-------------------------------------------------------------------
worker-0      0     2    4120 0.1.0    127.0.0.1:50321      1.40 ms
worker-1      1     2    9876 0.1.0    127.0.0.1:50322      1.28 ms
```

The numbers above are an example; yours will differ. The protocol is described in [docs/grpc-protocol.md](docs/grpc-protocol.md). Workers and the master can also be run by hand with `python -m grid.worker --rank 0 --world-size 2 --port 23461` and `python -m grid.master --workers 127.0.0.1:23461,127.0.0.1:23462` (set `PYTHONPATH=src` when running outside pytest).

## Weights and sharding

```
python scripts/inspect_weights.py --demo
```

This prints a tiny random Llama-style model and, for every tensor, how it is split across workers (column, row, vocab or replicated) and how many MB each worker holds. Weights are read with memory-mapping, so a worker loads only its own shard. Details, and the limits of the shard plan (for example why Qwen2.5-0.5B runs on 1 or 2 workers), are in [docs/weights.md](docs/weights.md).

## Tensor-parallel layers

```
python scripts/tp_check.py --workers 2
```

Two worker processes each hold half of a transformer-sized MLP (a column-parallel layer, a ReLU, then a row-parallel layer), exchange partial results with one `gloo` all-reduce, and the script checks the answer against the same layers computed on the full weights. It prints the largest difference, the weights each worker holds, and the all-reduce time. How the split works, and why a pair of layers needs only one all-reduce, is in [docs/tensor-parallel.md](docs/tensor-parallel.md).

## Tensor-parallel transformer block

```
python scripts/block_check.py --workers 2
```

A full MLP (SwiGLU) and attention block (rotary embeddings, grouped-query attention, key/value cache) with the real shape of a Qwen2.5-0.5B layer is split across worker processes. Each block needs one all-reduce, so a layer needs two. The script checks the sharded result against the same block on the full weights, with and without the cache, and prints what each worker holds. More in [docs/tensor-parallel.md](docs/tensor-parallel.md).

## Paged KV cache

```
python scripts/kv_demo.py
python scripts/kv_demo.py --check
```

Each worker keeps the keys and values of its own attention heads in a pool of fixed-size blocks, one block table per sequence. A block is taken only when the last one is full and is returned when the sequence finishes, so memory is fixed up front, never fragmented, and a full cache is refused cleanly. The first command draws the block map and the Qwen2.5-0.5B memory figures; the second checks cached decoding of two interleaved sequences on worker processes against full attention. More in [docs/kv-cache.md](docs/kv-cache.md).

## Continuous batching

```
python scripts/batch_demo.py
python scripts/batch_demo.py --check
```

The scheduler decides at every step which requests run together: a finished request leaves at once and a waiting one takes its place, instead of a group waiting for its slowest member. Requests are served first come, first served, and one is admitted only if the cache can hold its whole reply, so a running request can never run out of blocks. The demo draws static and continuous batching for the same eight requests; with a batch of 4, continuous needs 10 steps instead of 13. More in [docs/batching.md](docs/batching.md).

## Gateway

```
python scripts/serve.py
curl -X POST http://127.0.0.1:8765/v1/generate -H "Content-Type: application/json" -d "{\"prompt\": \"hello grid\", \"max_new_tokens\": 5}"
curl http://127.0.0.1:8765/stats
```

An engine loop runs the scheduler's steps and a FastAPI gateway sits in front of it: `POST /v1/generate`, `GET /stats` for the dashboard and `GET /health`. Many clients are served in the same batch, a full queue answers 503, a request that can never fit answers 413, and a failing step only fails the requests in it. Until the real model is connected, a simulated model with real cache traffic stands in for it. More in [docs/gateway.md](docs/gateway.md).

## Hardware and honesty notes

The development machine is a laptop with a 2-core CPU, 12 GB of RAM and no NVIDIA GPU. So:

- Tensor parallelism is developed and tested with the `gloo` backend on CPU, using small models (for example Qwen2.5-0.5B or TinyLlama-1.1B).
- The NCCL path is written behind a backend switch but **cannot be run on this hardware**. The final report will say so plainly.
- Performance numbers (time to first token, tokens per second, inter-node latency) are measured and reported as they are, with the hardware stated.

## Project layout (planned)

```
src/grid/      engine code: master, worker, gRPC services, tensor-parallel layers, KV cache, scheduler, gateway
proto/         gRPC service definitions
tests/         automated tests
docs/          architecture, plan, review checklist, final report
deploy/        Dockerfiles and the Docker Swarm stack file
scripts/       helper and benchmark scripts
```

## Branching

Feature branches (`feat/...`, `docs/...`, `fix/...`) merged into `main` with `--no-ff`. Commits are signed. Code that synchronises distributed memory goes through a written review ([docs/review-checklist.md](docs/review-checklist.md)) before it is merged.

## Credits

Designed and built by Vignesh Yadala.
