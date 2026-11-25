# Plan and review checklist

Source: Zaalima Development Pvt. Ltd. project document and SOP version 1.1 (Month 2 starts on 8 November).

## Weeks

| Week | Dates | Deliverables |
|------|-------|--------------|
| 1 | 8 to 15 Nov | Master-worker topology with asyncio and gRPC, safetensors weight loading with memory-mapping, commit and branch rules |
| 2 | 16 to 22 Nov | Tensor parallelism with `torch.distributed` (attention heads and feed-forward layers), asynchronous KV-cache manager, peer review of the memory-synchronisation code |
| 3 | 23 to 29 Nov | Continuous batching, NCCL backend switch, FastAPI + WebSocket gateway |
| 4 | 30 Nov to 6 Dec | Docker Swarm deployment, load tests (time to first token, inter-node latency), final report |

## Mid Review (end of week 2, between 23 and 30 Nov)

- [x] Network topology and master-worker communication operational
- [x] Tensor parallelism distributing weights correctly without data loss
- [ ] Branch commit rules and initial review cycles passed
- [ ] Commits on at least 10 different days in the 14 days before the review

## Final Review (between 7 and 15 Dec)

- [ ] Distributed inference engine generating tokens across multiple nodes
- [x] Continuous batching and KV-cache optimisation verified (scheduler and cache, on a simulated model)
- [ ] Final architectural report and deployment scripts finalised
- [ ] A commit on every one of the 20 days before the review (no gaps)
