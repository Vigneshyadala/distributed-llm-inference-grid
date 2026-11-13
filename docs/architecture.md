# Architecture (design intent)

This document describes the planned design. Sections are updated as each part is built and tested.

## Components

| Component | Responsibility |
|-----------|----------------|
| **Master** | Owns the request queue and the scheduler. Decides which sequences run in each generation step and tells the workers. |
| **Worker** | Holds one shard of the model weights and the matching slice of the KV cache. Runs its share of every layer. |
| **Gateway** | FastAPI service with WebSockets. Accepts prompts and streams tokens back to the client. |
| **gRPC layer** | Master-worker control and data messages, implemented with `grpc.aio` on asyncio. |
| **Collective backend** | `torch.distributed` all-reduce between workers. `gloo` on CPU, with a switch for `nccl` on NVIDIA GPUs. |

## Tensor parallelism

For each transformer layer the weights are split across N workers:

- **Attention:** the query, key and value projections are split by attention head (column parallel). Each worker attends with its own heads. The output projection is split by rows, and the partial results are summed with one all-reduce.
- **Feed-forward:** the up and gate projections are split by columns, the down projection by rows, again with one all-reduce.
- **Constraint:** the number of attention heads and key-value heads must divide evenly by the number of workers. The model is chosen with this in mind.

## Weight loading

Weights are read from HuggingFace `safetensors` files with memory-mapping, and each worker loads only its own slice, so no single node ever needs the full model in RAM.

## KV cache

Each worker keeps the keys and values for its own heads. An asynchronous manager hands out fixed-size blocks per sequence and frees them when the sequence ends, to avoid memory fragmentation during long generations.

## Continuous batching

The scheduler works at the level of single generation steps. Finished sequences leave the batch and waiting requests join it at the next step, so new requests do not wait for old ones to finish.

## Deployment

Docker images for the master, workers and gateway, deployed as a Docker Swarm stack with placement constraints for the workers. Development and tests run without Docker.
