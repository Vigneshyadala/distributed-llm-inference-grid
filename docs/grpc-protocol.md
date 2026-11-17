# Master-worker protocol (gRPC)

The service is defined in [`proto/grid/rpc/worker.proto`](../proto/grid/rpc/worker.proto). The master talks to every worker at the same time with `grpc.aio` and `asyncio.gather`.

## Calls

| Call | Request | Reply | Purpose |
|------|---------|-------|---------|
| `Hello` | `master_id` | `rank`, `world_size`, `worker_id`, `version`, `pid` | First contact: the worker says who it is |
| `Ping` | `payload` (bytes) | the same `payload` | Round trip used to measure inter-node latency |

## Topology check

After every `Hello` round the master verifies that the workers form one consistent grid, and refuses to continue otherwise:

- every worker reports the same `world_size`, and it equals the number of workers that answered;
- the ranks are exactly `0 .. N-1`, with no duplicates and no gaps.

An unreachable worker raises `WorkerUnavailable` with its address; the master never hangs, because every call has a timeout.

## Generated code

The `*_pb2.py` and `*_pb2_grpc.py` modules are not committed. `grid.rpc` rebuilds them with the installed `grpcio-tools` the first time it is imported (and again whenever the `.proto` file changes), so they always match the installed `grpcio`. To force a rebuild, delete `src/grid/rpc/worker_pb2.py` and `src/grid/rpc/worker_pb2_grpc.py`; the next import recreates them.

## Still to come

`Forward` (run this worker's shard of a layer), `AllReduce` coordination, KV-cache calls and token streaming are added in the next weeks and will be described here.
