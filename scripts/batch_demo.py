"""Compare static and continuous batching on the same requests.

    python scripts/batch_demo.py                 draw both schedules and print the numbers
    python scripts/batch_demo.py --check         also verify the results (exit code 0 means OK)
    python scripts/batch_demo.py --batch 2 --blocks 12     other batch size / cache size

Needs only numpy. A fake model stands in for the real one: it stores keys and values in the
real paged cache and answers a dummy token, so the schedule, the memory use and the freeing of
blocks are all real. One step takes one time unit, whatever the batch size, which is the point of
batching: serving more requests per step costs almost nothing extra.

In the pictures, one row per request and one column per step:
    P prefill (prompt goes in, first token comes out)    d decode (one more token)
    . waiting for a place                                 - holding a place with nothing to do
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid.kvcache import PagedKVCache  # noqa: E402
from grid.scheduler import Request, Scheduler, run_to_completion  # noqa: E402

# (name, prompt tokens, tokens to generate): short and long replies mixed, as in real traffic
WORKLOAD = [("A", 6, 3), ("B", 4, 8), ("C", 8, 2), ("D", 5, 6),
            ("E", 3, 4), ("F", 7, 2), ("G", 4, 5), ("H", 6, 3)]
KV_HEADS, HEAD_DIM, LAYERS, BLOCK_SIZE = 2, 4, 2, 4


def fake_model(cache):
    def run(plan):
        tokens = {}
        for request in plan.prefill + plan.decode:
            count = request.prompt_len if request in plan.prefill else 1
            for layer in range(cache.layers):
                data = np.zeros((1, KV_HEADS, count, HEAD_DIM), dtype=np.float32)
                cache.extend(request.id, layer, data, data)
            tokens[request.id] = 1
        return tokens
    return run


def continuous(batch, blocks):
    cache = PagedKVCache(LAYERS, KV_HEADS, HEAD_DIM, BLOCK_SIZE, blocks)
    scheduler = Scheduler(cache, max_batch_size=batch)
    for name, prompt, new in WORKLOAD:
        scheduler.submit(Request(name, prompt, new))
    run_to_completion(scheduler, fake_model(cache))
    rows = {}
    for record in scheduler.timeline:
        for name in record["prefill"]:
            rows.setdefault(name, {})[record["step"]] = "P"
        for name in record["decode"]:
            rows.setdefault(name, {})[record["step"]] = "d"
    steps = len(scheduler.timeline)
    lanes = {}
    for name, _, _ in WORKLOAD:
        request = scheduler.get(name)
        lanes[name] = "".join(
            rows.get(name, {}).get(s, "." if request.state != "rejected" and s < request.admitted_step else " ")
            for s in range(steps)) if request.admitted_step is not None else "rejected"
    return scheduler, cache, lanes, steps


def static(batch):
    """Groups of `batch` requests: a group runs until its longest request is done."""
    lanes, finish, clock = {}, {}, 0
    groups = [WORKLOAD[i:i + batch] for i in range(0, len(WORKLOAD), batch)]
    total = sum(max(new for _, _, new in group) for group in groups)
    for group in groups:
        length = max(new for _, _, new in group)
        for name, _, new in group:
            lanes[name] = "." * clock + "P" + "d" * (new - 1) + "-" * (length - new)
            finish[name] = clock + new
        clock += length
    return {n: lane.ljust(total) for n, lane in lanes.items()}, finish, total


def draw(title, lanes, steps):
    print(f"{title}   (steps 0-{steps - 1})")
    for name, _, _ in WORKLOAD:
        print(f"  {name} |{lanes[name].ljust(steps)}|")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare static and continuous batching.")
    parser.add_argument("--batch", type=int, default=4, help="most requests served at once")
    parser.add_argument("--blocks", type=int, default=40, help="blocks in the KV cache")
    parser.add_argument("--check", action="store_true", help="verify the results and set the exit code")
    args = parser.parse_args()

    scheduler, cache, c_lanes, c_steps = continuous(args.batch, args.blocks)
    s_lanes, s_finish, s_steps = static(args.batch)
    draw("STATIC batching: a group waits for its slowest member", s_lanes, s_steps)
    draw("CONTINUOUS batching: a finished request is replaced at once", c_lanes, c_steps)

    useful = sum(new for _, _, new in WORKLOAD)
    s_lat = sum(s_finish.values()) / len(WORKLOAD)
    done = scheduler.finished
    c_lat = sum(r.latency_steps for r in done) / len(done) if done else 0.0
    print(f"{len(WORKLOAD)} requests, batch of {args.batch}, cache of {args.blocks} blocks x {BLOCK_SIZE} tokens")
    print(f"{'':12}{'steps':>7}{'average latency':>18}{'slots doing work':>19}")
    print(f"{'static':12}{s_steps:>7}{s_lat:>14.1f} st{useful / (args.batch * s_steps):>17.0%}")
    print(f"{'continuous':12}{c_steps:>7}{c_lat:>14.1f} st{useful / (args.batch * c_steps):>17.0%}")
    print(f"peak blocks reserved {max(t['reserved_blocks'] for t in scheduler.timeline)} of {args.blocks}, "
          f"blocks free at the end {cache.free_blocks}")

    if args.check:
        problems = []
        if len(done) != len(WORKLOAD):
            problems.append(f"only {len(done)} of {len(WORKLOAD)} requests finished "
                            f"({len(scheduler.rejected)} rejected: the cache may be too small)")
        if cache.free_blocks != cache.num_blocks:
            problems.append("blocks were not all given back")
        if c_steps > s_steps:
            problems.append(f"continuous took more steps ({c_steps}) than static ({s_steps})")
        if done and c_lat > s_lat:
            problems.append(f"continuous latency {c_lat:.1f} is worse than static {s_lat:.1f}")
        peak = max(t["running"] for t in scheduler.timeline)
        if problems and peak < min(args.batch, len(WORKLOAD)):
            problems.append(f"the cache only held {peak} requests at a time, so it limited the batch; "
                            f"try more --blocks")
        if problems:
            print("BATCH CHECK FAILED: " + "; ".join(problems))
            return 1
        print(f"BATCH CHECK OK: continuous finished in {c_steps} steps against {s_steps}, "
              f"with lower average latency ({c_lat:.1f} against {s_lat:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
