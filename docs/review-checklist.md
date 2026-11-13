# Review checklist for distributed-memory code

Used for every change that touches weight sharding, collectives or the KV cache. Fill it in on the pull request before merging.

- [ ] Shapes: every sharded tensor has the expected shape on every worker
- [ ] No data loss: the sharded result matches the single-process reference within a stated tolerance (test attached)
- [ ] Every collective call is made by all workers in the same order (no deadlock paths)
- [ ] Errors: a worker failure produces a clear error, not a hang (timeouts set)
- [ ] KV cache: blocks are always freed, including when a request is cancelled
- [ ] Memory: peak RAM per worker measured and noted
- [ ] Tests added or updated, and all tests pass
- [ ] Docs updated
