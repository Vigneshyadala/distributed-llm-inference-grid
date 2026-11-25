"""Record a real run of the engine to a file, so the dashboard can replay it without a server.

    python scripts/record_run.py                          writes src/grid/web/recorded.json
    python scripts/record_run.py --out my_run.json

Five requests arrive one after another at a small, 12-block cache (48 tokens), plus one that can
never fit. Watch for the third and fourth: one has to wait until the cache has room, and one is
refused. Everything in the file is what the engine really did; only the time offsets are kept
(as "dt", in milliseconds) so the dashboard can replay it at the original speed.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid.gateway import build_engine  # noqa: E402

BLOCKS, BLOCK_SIZE, BATCH, STEP_SECONDS = 12, 4, 4, 0.2
# (seconds after the start, prompt tokens, tokens to generate)
ARRIVALS = [(0.0, 6, 10), (0.5, 3, 6), (1.0, 9, 8), (1.5, 4, 12), (2.0, 30, 30), (3.0, 2, 5)]


async def record():
    engine = build_engine(blocks=BLOCKS, block_size=BLOCK_SIZE, max_batch_size=BATCH, step_seconds=STEP_SECONDS)
    await engine.start()
    scheduler = engine.scheduler
    queue = engine.subscribe(maxsize=10000)
    events = [{"type": "snapshot", "stats": scheduler.stats(), "cache": scheduler.cache.stats(),
               "recent_steps": [], "block_map": engine.block_map(), "t": 0}]

    async def client(delay, prompt, new):
        await asyncio.sleep(delay)
        await engine.submit(prompt, new).result()

    await asyncio.gather(*(client(*a) for a in ARRIVALS))
    await asyncio.sleep(0.3)
    while not queue.empty():
        events.append(queue.get_nowait())
    await engine.stop()
    start = events[1]["t"]
    for event in events:
        event["dt"] = round(1000 * (event.pop("t") - start)) if event["type"] != "snapshot" else 0
    return {"meta": {"blocks": BLOCKS, "block_size": BLOCK_SIZE, "max_batch": BATCH, "step_seconds": STEP_SECONDS},
            "events": events}


def main():
    parser = argparse.ArgumentParser(description="Record an engine run for the dashboard's replay mode.")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "src" / "grid" / "web" / "recorded.json"))
    args = parser.parse_args()
    data = asyncio.run(record())
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    steps = [e for e in data["events"] if e["type"] == "step"]
    print(f"recorded {len(data['events'])} events ({len(steps)} steps, "
          f"{data['events'][-1]['dt'] / 1000:.1f} s) to {args.out}")


if __name__ == "__main__":
    main()
