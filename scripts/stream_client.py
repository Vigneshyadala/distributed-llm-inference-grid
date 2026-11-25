"""Watch the gateway live from a terminal.

    python scripts/stream_client.py "hello grid" --max-new 20     stream one reply, token by token
    python scripts/stream_client.py --events                       follow every step the engine runs
    python scripts/stream_client.py --url ws://127.0.0.1:23456 --events

Start the server first (python scripts/serve.py). Run several clients at once, then follow the
events in a third window, to see requests join and leave the same batch.
"""
import argparse
import json
import sys
import time


def format_event(event):
    """One readable line for an engine event (None for events that need no line)."""
    kind = event.get("type")
    if kind == "snapshot":
        stats = event["stats"]
        return (f"connected: step {stats['step']}, {stats['running']} running, {stats['waiting']} waiting, "
                f"{stats['blocks_free']}/{stats['blocks_total']} blocks free")
    if kind == "submitted":
        return f"+ {event['id']} submitted ({event['prompt_len']} prompt tokens, up to {event['max_new_tokens']} new)"
    if kind == "rejected":
        return f"x {event['id']} rejected: {event['reason']}"
    if kind == "step":
        stats = event["stats"]
        prefill = ",".join(event["prefill"]) or "-"
        decode = ",".join(event["decode"]) or "-"
        return (f"step {event['step']:>4}  prefill [{prefill}]  decode [{decode}]  running {event['running']}"
                f"  waiting {event['waiting']}  blocks {event['blocks_used']}/{stats['blocks_total']}"
                f"  {event['ms']:.1f} ms")
    if kind == "finished":
        return (f"- {event['id']} {event['kind']} ({event['reason']}) after {event['tokens']} tokens, "
                f"waited {event['wait_steps']} steps, latency {event['latency_steps']} steps")
    if kind == "stopped":
        return "the engine stopped"
    return None


def stream_reply(url, prompt, max_new):
    from websockets.sync.client import connect

    start = time.perf_counter()
    with connect(f"{url}/v1/stream") as ws:
        ws.send(json.dumps({"prompt": prompt, "max_new_tokens": max_new}))
        while True:
            event = json.loads(ws.recv())
            elapsed = 1000 * (time.perf_counter() - start)
            if event["type"] == "accepted":
                print(f"{elapsed:8.1f} ms  accepted as {event['id']} ({event['prompt_tokens']} prompt tokens)")
            elif event["type"] == "token":
                print(f"{elapsed:8.1f} ms  token {event['index']:>3} = {event['token']}")
            else:
                print(f"{elapsed:8.1f} ms  {event['type']}" + (f": {event.get('reason') or event.get('detail')}"
                                                              if event["type"] != "done" else f" ({event['reason']})"))
                return 0 if event["type"] == "done" else 1


def follow_events(url):
    from websockets.sync.client import connect

    with connect(f"{url}/ws/events") as ws:
        try:
            while True:
                line = format_event(json.loads(ws.recv()))
                if line:
                    print(line, flush=True)
        except KeyboardInterrupt:
            return 0


def main():
    parser = argparse.ArgumentParser(description="Stream a reply or follow the engine's events.")
    parser.add_argument("prompt", nargs="?", help="the prompt to send (omit with --events)")
    parser.add_argument("--max-new", type=int, default=20)
    parser.add_argument("--events", action="store_true", help="follow every step instead of sending a prompt")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    args = parser.parse_args()
    try:
        if args.events:
            return follow_events(args.url)
        if not args.prompt:
            parser.error("give a prompt, or use --events")
        return stream_reply(args.url, args.prompt, args.max_new)
    except ImportError:
        print("this needs the websockets package: pip install websockets")
        return 2
    except OSError as exc:
        print(f"could not reach {args.url}: {exc}. Is the server running (python scripts/serve.py)?")
        return 2


if __name__ == "__main__":
    sys.exit(main())
