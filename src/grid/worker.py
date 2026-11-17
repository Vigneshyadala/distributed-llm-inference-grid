"""Worker node: one process that will own one shard of the model.

For now a worker can only introduce itself (Hello) and echo a payload (Ping).
Weight shards, tensor-parallel layers and the KV cache are added in later commits.
"""
import argparse
import asyncio
import os

import grpc

import grid
from grid.rpc import worker_pb2, worker_pb2_grpc


class WorkerService(worker_pb2_grpc.WorkerServicer):
    def __init__(self, rank, world_size):
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside a grid of {world_size} workers")
        self.rank = rank
        self.world_size = world_size
        self.worker_id = f"worker-{rank}"

    async def Hello(self, request, context):
        return worker_pb2.HelloReply(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            version=grid.__version__,
            pid=os.getpid(),
        )

    async def Ping(self, request, context):
        return worker_pb2.PingReply(payload=request.payload)


async def start_worker(rank, world_size, host="127.0.0.1", port=0):
    """Start a worker server. Port 0 picks a free port. Returns (server, bound_port)."""
    service = WorkerService(rank, world_size)
    server = grpc.aio.server()
    worker_pb2_grpc.add_WorkerServicer_to_server(service, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if not bound:
        raise OSError(f"could not bind {host}:{port}")
    await server.start()
    return server, bound


async def serve(rank, world_size, host, port):
    server, bound = await start_worker(rank, world_size, host, port)
    print(f"worker-{rank} (rank {rank} of {world_size}) listening on {host}:{bound}", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(1)


def main():
    parser = argparse.ArgumentParser(description="Run one worker node.")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    args = parser.parse_args()
    try:
        asyncio.run(serve(args.rank, args.world_size, args.host, args.port))
    except KeyboardInterrupt:
        print("worker stopped")


if __name__ == "__main__":
    main()
