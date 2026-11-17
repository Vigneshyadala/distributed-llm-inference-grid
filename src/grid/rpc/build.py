"""Generate the gRPC Python code from the .proto files in proto/.

The generated *_pb2 modules are not committed. They are rebuilt on demand, with
the grpcio-tools version installed on the machine, so they always match the
installed grpcio.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "src"
PROTO_NAME = "grid/rpc/worker.proto"
PROTO_FILE = PROTO_DIR / PROTO_NAME
GENERATED = [
    OUT_DIR / "grid" / "rpc" / "worker_pb2.py",
    OUT_DIR / "grid" / "rpc" / "worker_pb2_grpc.py",
]


def is_current():
    """True when the generated files exist and are newer than the .proto file."""
    if not all(path.exists() for path in GENERATED):
        return False
    newest_proto = PROTO_FILE.stat().st_mtime
    return all(path.stat().st_mtime >= newest_proto for path in GENERATED)


def generate(force=False):
    """Run protoc. Returns True if it generated code, False if it was already current."""
    if is_current() and not force:
        return False
    from grpc_tools import protoc

    code = protoc.main(
        [
            "protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            PROTO_NAME,
        ]
    )
    if code != 0:
        raise RuntimeError(f"protoc failed with exit code {code}")
    return True

