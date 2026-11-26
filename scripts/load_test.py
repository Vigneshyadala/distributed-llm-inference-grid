"""Load-test the gateway: many clients at once, then a short report.

    python scripts/serve.py --port 23456 --step-seconds 0.02        (in one terminal)
    python scripts/load_test.py --url ws://127.0.0.1:23456 --clients 32 --max-new 24

Every client opens /v1/stream, sends one prompt and reads tokens until the reply ends. The report
gives, for the whole run, requests served or refused, tokens per second, and p50/p95/max for
time to first token (TTFT) and for the full reply. Times are measured at the client, so they
include the network and the gateway as well as the engine.

Until the real model is connected, the engine runs a simulated model: these numbers measure the
scheduler, the cache and the gateway, not language-model speed.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def percentile(values, q):
    """The q-th percentile (0 to 100) by linear interpolation; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def one_client(url, prompt, max_new):
    """Run one streamed request. Returns a dict; 'ok' is False for any refusal or failure."""
    from websockets.sync.client import connect

    start = time.perf_counter()
    first = None
    tokens = 0
    try:
        with connect(f"{url}/v1/stream") as ws:
            ws.send(json.dumps({"prompt": prompt, "max_new_tokens": max_new}))
            while True:
                event = json.loads(ws.recv())
                kind = event.get("type")
                if kind == "token":
                    tokens += 1
                    if first is None:
                        first = time.perf_counter() - start
                elif kind == "done":
                    return {"ok": True, "tokens": tokens, "ttft": first,
                            "total": time.perf_counter() - start, "reason": event.get("reason")}
                elif kind in ("error", "rejected", "cancelled"):
                    return {"ok": False, "tokens": tokens, "ttft": first, "total": time.perf_counter() - start,
                            "reason": str(event.get("detail") or event.get("reason") or kind)}
    except Exception as exc:        # connection refused, closed early, and so on
        return {"ok": False, "tokens": tokens, "ttft": first, "total": time.perf_counter() - start,
                "reason": f"{type(exc).__name__}: {exc}"}


def summarise(results, wall):
    """Turn per-client results into the numbers in the report."""
    served = [r for r in results if r["ok"]]
    tokens = sum(r["tokens"] for r in served)
    ttft = [1000 * r["ttft"] for r in served if r["ttft"] is not None]
    total = [1000 * r["total"] for r in served]
    reasons = {}
    for r in results:
        if not r["ok"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    def stats(values):
        return {"p50": percentile(values, 50), "p95": percentile(values, 95), "max": max(values) if values else None}

    return {"requests": len(results), "served": len(served), "refused": len(results) - len(served),
            "refusal_reasons": reasons, "tokens": tokens, "wall_seconds": wall,
            "tokens_per_second": tokens / wall if wall > 0 else 0.0,
            "ttft_ms": stats(ttft), "total_ms": stats(total)}


def run(url, clients, concurrency, prompt, max_new):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_client, url, f"{prompt} {i}", max_new) for i in range(clients)]
        results = [f.result() for f in futures]
    return summarise(results, time.perf_counter() - start)


def format_report(s):
    def ms(v):
        return "-" if v is None else f"{v:8.1f} ms"

    lines = [f"requests   {s['requests']}  served {s['served']}  refused {s['refused']}",
             f"tokens     {s['tokens']} in {s['wall_seconds']:.2f} s  =  {s['tokens_per_second']:.1f} tokens/s",
             "           " + "      p50          p95          max",
             f"TTFT       {ms(s['ttft_ms']['p50'])} {ms(s['ttft_ms']['p95'])} {ms(s['ttft_ms']['max'])}",
             f"full reply {ms(s['total_ms']['p50'])} {ms(s['total_ms']['p95'])} {ms(s['total_ms']['max'])}"]
    for reason, count in s["refusal_reasons"].items():
        lines.append(f"refused    {count} x {reason}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Load-test the gateway with many streamed requests.")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--clients", type=int, default=32, help="total requests to send")
    parser.add_argument("--concurrency", type=int, default=None, help="requests in flight at once (default: all)")
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--prompt", default="load test prompt")
    parser.add_argument("--json", metavar="FILE", help="also write the numbers to this file")
    args = parser.parse_args()

    summary = run(args.url, args.clients, args.concurrency or args.clients, args.prompt, args.max_new)
    print(format_report(summary))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return 0 if summary["served"] == summary["requests"] else 1


if __name__ == "__main__":
    sys.exit(main())
