# Distributed Memory Grid for Real-Time LLM Inference

A custom distributed inference engine in Python. It shards a Large Language Model across several worker nodes with **tensor parallelism**, talks between nodes over **gRPC and asyncio**, manages a **KV cache** per worker and serves tokens to clients through a **FastAPI + WebSocket** gateway with **continuous batching**.

> Enterprise Project 1 of the Zaalima Development Pvt. Ltd. internship program (Month 2). Built by Vignesh Yadala.

## Status

| Week | Scope | State |
|------|-------|-------|
| 1 | Master-worker topology (asyncio + gRPC), safetensors weight loading with memory-mapping, commit rules | In progress |
| 2 | Tensor parallelism with `torch.distributed`, asynchronous KV-cache manager, review cycles | Planned |
| 3 | Continuous batching, collective-communication backend switch, FastAPI + WebSocket gateway | Planned |
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
