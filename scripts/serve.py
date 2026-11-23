"""Start the gateway on the simulated model.

    python scripts/serve.py                      http://127.0.0.1:8765
    python scripts/serve.py --port 23456 --step-seconds 0.05

Then, in another terminal:
    curl http://127.0.0.1:8765/health
    curl -X POST http://127.0.0.1:8765/v1/generate -H "Content-Type: application/json" -d "{\"prompt\": \"hello grid\", \"max_new_tokens\": 5}"
    curl http://127.0.0.1:8765/stats
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grid.gateway import build_engine, create_app  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the gateway on the simulated model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--blocks", type=int, default=256, help="blocks in the KV cache")
    parser.add_argument("--batch", type=int, default=8, help="most requests served at once")
    parser.add_argument("--max-waiting", type=int, default=64, help="queue limit before answering 503")
    parser.add_argument("--step-seconds", type=float, default=0.02, help="simulated time per step")
    args = parser.parse_args()

    import uvicorn

    engine = build_engine(blocks=args.blocks, max_batch_size=args.batch,
                          max_waiting=args.max_waiting, step_seconds=args.step_seconds)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
