# Load test

`scripts/load_test.py` sends many streamed requests to the gateway at once and prints a short report.
It is the first piece of the Week 4 load tests (time to first token now, inter-node latency once
the real workers run).

## Run it

```
python scripts/serve.py --port 23456 --step-seconds 0.02
python scripts/load_test.py --url ws://127.0.0.1:23456 --clients 32 --max-new 24
```

Options: `--clients` (total requests), `--concurrency` (how many are in flight at once, default all),
`--max-new`, `--prompt`, and `--json FILE` to also save the numbers. The exit code is 0 only if every
request was served.

## What the report means

| Line | Meaning |
|------|---------|
| requests | how many were sent, served and refused (a full queue answers 503, an oversized request 413) |
| tokens | tokens received, the wall-clock time of the whole run, and tokens per second over that time |
| TTFT | time to first token: from sending the prompt to the first token arriving at the client |
| full reply | from sending the prompt to the last token |

Each time column shows p50 (the typical request), p95 (the slow tail) and the maximum.
Times are measured at the client, so they include the network and the gateway as well as the engine.

## Reading the numbers

The engine serves at most `--batch` requests at a time (8 by default). With 32 clients at once, the
first 8 start immediately and the rest wait for a place in the batch. That wait is part of their TTFT,
which is why p95 TTFT is much larger than p50 in a run like this. It is queueing, not slowness of a
single request. Change `--batch` on the server and run again to see the effect.

## Honest limits

- The engine runs the simulated model for now (`docs/gateway.md`). These numbers measure the scheduler,
  the paged cache and the gateway. They say nothing about language-model speed; that comes when the
  real model is connected.
- The load generator runs on the same machine as the server, so they share the CPU. Treat the numbers as a
  comparison between runs on one machine, not as absolute figures.
- Run the same command a few times; the first run is usually the slowest.
