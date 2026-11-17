"""Start N worker processes on this machine and let the master greet them.

    python scripts/hello_grid.py                 two workers on free ports
    python scripts/hello_grid.py --workers 3
"""
import argparse
import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)

from grid.master import Master, WorkerUnavailable, format_table  # noqa: E402


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_until_ready(addresses, timeout):
    """Greet the workers, retrying until they are all up or the time runs out."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            async with Master(addresses, timeout=1.0) as master:
                infos = await master.hello()
                return infos, await master.ping()
        except WorkerUnavailable:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description="Start local workers and greet them.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    args = parser.parse_args()

    ports = [free_port() for _ in range(args.workers)]
    addresses = [f"127.0.0.1:{port}" for port in ports]
    env = {**os.environ, "PYTHONPATH": SRC}
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "grid.worker", "--rank", str(rank),
             "--world-size", str(args.workers), "--port", str(port)],
            env=env,
            stdout=subprocess.DEVNULL,
        )
        for rank, port in enumerate(ports)
    ]
    try:
        infos, rtts = asyncio.run(wait_until_ready(addresses, args.startup_timeout))
        print(format_table(infos, rtts))
        distinct = len({info.pid for info in infos})
        print(f"\ngrid ready: {len(infos)} workers in {distinct} distinct processes")
        return 0 if distinct == len(infos) else 1
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())
