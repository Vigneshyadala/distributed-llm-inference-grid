import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stream_client", ROOT / "scripts" / "stream_client.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

STATS = {"blocks_total": 256, "step": 9, "running": 2, "waiting": 1, "blocks_free": 250}


def test_step_line_shows_who_prefills_and_decodes():
    line = client.format_event({"type": "step", "step": 7, "prefill": ["req-3"], "decode": ["req-1", "req-2"],
                                "running": 3, "waiting": 0, "blocks_used": 6, "ms": 20.04, "stats": STATS})
    assert line == ("step    7  prefill [req-3]  decode [req-1,req-2]  running 3  waiting 0  blocks 6/256  20.0 ms")


def test_empty_prefill_or_decode_is_a_dash():
    line = client.format_event({"type": "step", "step": 1, "prefill": [], "decode": ["a"], "running": 1,
                                "waiting": 0, "blocks_used": 1, "ms": 1.0, "stats": STATS})
    assert "prefill [-]" in line and "decode [a]" in line


def test_lifecycle_lines():
    assert client.format_event({"type": "snapshot", "stats": STATS}) == \
        "connected: step 9, 2 running, 1 waiting, 250/256 blocks free"
    assert client.format_event({"type": "submitted", "id": "req-1", "prompt_len": 4, "max_new_tokens": 9}) == \
        "+ req-1 submitted (4 prompt tokens, up to 9 new)"
    assert client.format_event({"type": "rejected", "id": "req-2", "reason": "too big"}) == "x req-2 rejected: too big"
    assert client.format_event({"type": "finished", "id": "req-1", "kind": "done", "reason": "length", "tokens": 9,
                                "wait_steps": 0, "latency_steps": 9}) == \
        "- req-1 done (length) after 9 tokens, waited 0 steps, latency 9 steps"
    assert client.format_event({"type": "stopped"}) == "the engine stopped"
    assert client.format_event({"type": "something-new"}) is None
