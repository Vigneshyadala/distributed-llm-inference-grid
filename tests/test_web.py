import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from grid.gateway import WEB_DIR, build_engine, create_app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def client(**kw):
    return TestClient(create_app(build_engine(**kw)))


def test_the_dashboard_page_and_its_files_are_served():
    with client() as c:
        page = c.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "Distributed Memory Grid" in page.text and "assets/app.js" in page.text
        for name, kind in (("app.js", "javascript"), ("style.css", "css"), ("recorded.json", "json")):
            reply = c.get(f"/assets/{name}")
            assert reply.status_code == 200 and kind in reply.headers["content-type"], name
        assert c.get("/assets/nothing.js").status_code == 404
        assert c.get("/health").status_code == 200                      # the API still works beside it


def test_the_page_can_be_left_out():
    with TestClient(create_app(build_engine(), web_dir=None)) as c:
        assert c.get("/").status_code == 404 and c.get("/health").status_code == 200


def test_every_page_asset_the_html_names_exists():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for name in ("style.css", "app.js"):
        assert f"assets/{name}" in html and (WEB_DIR / name).exists()


def test_snapshot_stats_and_step_events_carry_the_block_map():
    with client(blocks=16, block_size=4) as c:
        assert c.get("/stats").json()["block_map"] == [None] * 16
        with c.websocket_connect("/ws/events") as feed:
            snapshot = feed.receive_json()
            assert snapshot["block_map"] == [None] * 16 and snapshot["cache"]["block_size"] == 4
            c.post("/v1/generate", json={"prompt": "a b c d e", "max_new_tokens": 3})
            steps = []
            while True:
                event = feed.receive_json()
                if event["type"] == "step":
                    steps.append(event)
                if event["type"] == "finished":
                    break
    first = steps[0]["block_map"]
    assert len(first) == 16 and first[0] == ["req-1", 4] and first[1] == ["req-1", 1] and first[2] is None
    assert steps[1]["block_map"][:2] == [["req-1", 4], ["req-1", 2]]            # 5 prompt tokens + 1 decoded
    assert steps[-1]["block_map"] == [None] * 16                               # it finished: blocks are back


def test_the_recorded_run_is_a_consistent_real_run():
    data = json.loads((WEB_DIR / "recorded.json").read_text(encoding="utf-8"))
    meta, events = data["meta"], data["events"]
    assert events[0]["type"] == "snapshot" and len(events[0]["block_map"]) == meta["blocks"]
    assert [e["dt"] for e in events] == sorted(e["dt"] for e in events)
    submitted = {e["id"] for e in events if e["type"] == "submitted"}
    steps = [e for e in events if e["type"] == "step"]
    assert steps and all(len(e["block_map"]) == meta["blocks"] for e in steps)
    assert [e["step"] for e in steps] == list(range(len(steps)))
    for e in steps:
        assert set(e["prefill"]) | set(e["decode"]) <= submitted
        assert e["running"] <= meta["max_batch"] and e["blocks_used"] <= meta["blocks"]
    kinds = [e["type"] for e in events]
    assert "rejected" in kinds and kinds.count("finished") >= 4          # one refused, the rest served
    assert max(e["waiting"] for e in steps) >= 1                         # someone had to wait for room
    assert steps[-1]["block_map"] == [None] * meta["blocks"]             # every block given back


def test_record_run_writes_a_replayable_file(tmp_path):
    out = tmp_path / "run.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "record_run.py"), "--out", str(out)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["events"][0]["type"] == "snapshot" and any(e["type"] == "step" for e in data["events"])
    assert "recorded" in result.stdout and "steps" in result.stdout
