import socket

import pytest

from grid.master import Master, TopologyError, WorkerUnavailable, format_table
from grid.worker import WorkerService, start_worker


@pytest.fixture
async def grid():
    """Start workers inside this process; returns their addresses."""
    started = []

    async def start(world_size, ranks=None):
        ranks = list(range(world_size)) if ranks is None else ranks
        addresses = []
        for rank in ranks:
            server, port = await start_worker(rank, world_size)
            started.append(server)
            addresses.append(f"127.0.0.1:{port}")
        return addresses

    yield start
    for server in started:
        await server.stop(None)


def closed_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def test_hello_reports_rank_world_size_and_version(grid):
    addresses = await grid(world_size=1)
    async with Master(addresses) as master:
        (info,) = await master.hello()
    assert (info.rank, info.world_size, info.worker_id) == (0, 1, "worker-0")
    assert info.version
    assert info.address == addresses[0]


async def test_hello_sorts_workers_by_rank_even_if_started_out_of_order(grid):
    addresses = await grid(world_size=3, ranks=[2, 0, 1])
    async with Master(addresses) as master:
        infos = await master.hello()
    assert [info.rank for info in infos] == [0, 1, 2]
    assert [info.worker_id for info in infos] == ["worker-0", "worker-1", "worker-2"]


async def test_ping_echoes_payload_and_measures_every_worker(grid):
    addresses = await grid(world_size=2)
    async with Master(addresses) as master:
        rtts = await master.ping(b"hello grid")
    assert set(rtts) == set(addresses)
    assert all(ms > 0 for ms in rtts.values())


async def test_unreachable_worker_is_reported_with_its_address(grid):
    addresses = await grid(world_size=2)
    dead = f"127.0.0.1:{closed_port()}"
    async with Master([addresses[0], dead], timeout=2.0) as master:
        with pytest.raises(WorkerUnavailable) as error:
            await master.hello()
    assert dead in str(error.value)
    assert addresses[0] not in str(error.value)


async def test_duplicate_ranks_are_rejected(grid):
    addresses = await grid(world_size=2, ranks=[0, 0])
    async with Master(addresses) as master:
        with pytest.raises(TopologyError):
            await master.hello()


async def test_world_size_must_match_the_number_of_workers(grid):
    addresses = await grid(world_size=3, ranks=[0, 1])
    async with Master(addresses) as master:
        with pytest.raises(TopologyError):
            await master.hello()


def test_worker_rejects_a_rank_outside_the_grid():
    with pytest.raises(ValueError):
        WorkerService(rank=2, world_size=2)
    with pytest.raises(ValueError):
        WorkerService(rank=0, world_size=0)


def test_master_needs_at_least_one_address():
    with pytest.raises(ValueError):
        Master([])


async def test_table_lists_every_worker(grid):
    addresses = await grid(world_size=2)
    async with Master(addresses) as master:
        infos = await master.hello()
        rtts = await master.ping()
    table = format_table(infos, rtts)
    assert "worker-0" in table and "worker-1" in table
    assert table.count(" ms") == 2
