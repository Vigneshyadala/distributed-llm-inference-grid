"""gRPC layer: service definitions live in proto/, the Python code is generated from them."""
from grid.rpc.build import generate

generate()  # does nothing once the generated modules are current
