"""Master node: connects to every worker over gRPC (asyncio) and checks the grid.

    python -m grid.master --workers 127.0.0.1:23461,127.0.0.1:23462
"""
import argparse
import asyncio
import time
from dataclasses import dataclass

import grpc

from grid.rpc import worker_pb2, worker_pb2_grpc


class WorkerError(RuntimeError):
    """A worker answered with an error."""


class WorkerUnavailable(WorkerError):
    """A worker could not be reached in time."""


class TopologyError(RuntimeError):
    """The workers do not form a consistent grid."""


@dataclass(frozen=True)
class WorkerInfo:
    address: str
    rank: int
    world_size: int
    worker_id: str
    version: str
    pid: int


def check_topology(infos):
    """Raise TopologyError unless the workers form ranks 0..N-1 of one grid of size N."""
    sizes = {info.world_size for info in infos}
    if sizes != {len(infos)}:
        raise TopologyError(
            f"workers report world_size {sorted(sizes)} but {len(infos)} workers answered"
        )
    ranks = sorted(info.rank for info in infos)
    if ranks != list(range(len(infos))):
        raise TopologyError(f"ranks must be exactly 0..{len(infos) - 1}, got {ranks}")


class Master:
    def __init__(self, addresses, master_id="master", timeout=5.0):
        if not addresses:
            raise ValueError("at least one worker address is required")
        self.addresses = list(addresses)
        self.master_id = master_id
        self.timeout = timeout
        self._channels = {}
        self._stubs = {}

    async def __aenter__(self):
        for address in self.addresses:
            channel = grpc.aio.insecure_channel(address)
            self._channels[address] = channel
            self._stubs[address] = worker_pb2_grpc.WorkerStub(channel)
        return self

    async def __aexit__(self, *exc_info):
        await asyncio.gather(*(channel.close() for channel in self._channels.values()))
        self._channels.clear()
        self._stubs.clear()

    async def _call(self, address, method, request):
        try:
            return await getattr(self._stubs[address], method)(request, timeout=self.timeout)
        except grpc.aio.AioRpcError as exc:
            reason = f"{address}: {exc.code().name}"
            if exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                raise WorkerUnavailable(reason) from exc
            raise WorkerError(reason) from exc

    async def _gather(self, coroutines):
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            if not all(isinstance(failure, WorkerError) for failure in failures):
                raise next(f for f in failures if not isinstance(f, WorkerError))
            kind = WorkerUnavailable if all(isinstance(f, WorkerUnavailable) for f in failures) else WorkerError
            raise kind("; ".join(str(failure) for failure in failures))
        return results

    async def hello(self):
        """Greet every worker at the same time. Returns WorkerInfo objects sorted by rank."""

        async def one(address):
            reply = await self._call(address, "Hello", worker_pb2.HelloRequest(master_id=self.master_id))
            return WorkerInfo(address, reply.rank, reply.world_size, reply.worker_id, reply.version, reply.pid)

        infos = await self._gather([one(address) for address in self.addresses])
        check_topology(infos)
        return sorted(infos, key=lambda info: info.rank)

    async def ping(self, payload=b"x" * 64):
        """Round-trip time to every worker in milliseconds, keyed by address."""

        async def one(address):
            started = time.perf_counter()
            reply = await self._call(address, "Ping", worker_pb2.PingRequest(payload=payload))
            elapsed_ms = (time.perf_counter() - started) * 1000
            if reply.payload != payload:
                raise WorkerError(f"{address}: ping payload came back changed")
            return address, elapsed_ms

        return dict(await self._gather([one(address) for address in self.addresses]))


def format_table(infos, rtts=None):
    """A small text table of the grid, one row per worker."""
    rtts = rtts or {}
    header = f"{'worker':<10} {'rank':>4} {'world':>5} {'pid':>7} {'version':<8} {'address':<18} {'ping':>9}"
    lines = [header, "-" * len(header)]
    for info in infos:
        rtt = rtts.get(info.address)
        ping = f"{rtt:.2f} ms" if rtt is not None else "-"
        lines.append(
            f"{info.worker_id:<10} {info.rank:>4} {info.world_size:>5} {info.pid:>7} "
            f"{info.version:<8} {info.address:<18} {ping:>9}"
        )
    return "\n".join(lines)


async def run(addresses, timeout):
    async with Master(addresses, timeout=timeout) as master:
        infos = await master.hello()
        rtts = await master.ping()
    print(format_table(infos, rtts))


def main():
    parser = argparse.ArgumentParser(description="Greet every worker and check the grid.")
    parser.add_argument("--workers", required=True, help="comma-separated host:port list")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    addresses = [item.strip() for item in args.workers.split(",") if item.strip()]
    try:
        asyncio.run(run(addresses, args.timeout))
    except (WorkerError, TopologyError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
