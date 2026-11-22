# Continuous batching

`src/grid/scheduler.py` decides, at every step, which requests run together. It runs no model: it
plans a step, the caller runs it, and the caller reports the new tokens. That keeps the policy
testable without torch and lets the same scheduler drive a fake model, one worker or the whole grid.

## Why batch at all

A model step costs about the same for one sequence or for eight, because the weights are read once
either way. Serving many requests per step is therefore almost free. The only question is how to
form the batch.

```
static       A A A A A A A A      a group runs until its longest member is done:
             B B B - - - - -      B is finished but its place stays idle,
             C C C C C C - -      and the next group cannot start

continuous   A A A A A A A A      a finished request leaves at once,
             B B B C C C C C      and the next waiting one takes its place on the very next step
```

`python scripts/batch_demo.py` draws both for the same eight requests and prints the numbers. With
a batch of 4 it needs 10 steps instead of 13, the average request finishes sooner (6.1 steps
instead of 8.1), and 82% of the batch slots do work instead of 63%.

## One step

Each running request does one of two things per step:

| Kind | What goes in | What comes out |
|------|--------------|----------------|
| prefill | the whole prompt | the cache fills, and the first token |
| decode | the previous token | one more token (earlier keys and values come from the paged cache) |

```python
scheduler = Scheduler(cache, max_batch_size=4)
scheduler.submit(Request("r1", prompt_len=12, max_new_tokens=40))
while not scheduler.idle:
    plan = scheduler.next_step()          # plan.prefill and plan.decode
    tokens = run_the_model(plan)          # {request id: new token}
    finished = scheduler.complete_step(plan, tokens)
```

`run_to_completion(scheduler, execute, arrivals)` is this loop, with optional arrivals at given steps.

## Admission policy

- **First come, first served.** A request is never overtaken, so a large one cannot starve behind
  a stream of small ones.
- **Reserve, do not guess.** A request is admitted only if the cache can hold its prompt and every
  token it may still generate, counting what running requests have reserved. A running request can
  therefore never hit `OutOfBlocks` halfway through a reply, and nothing has to be thrown out and
  recomputed. The price is that room is held a little longer than strictly needed.
- **Impossible requests are refused at once.** One that would not fit even in an empty cache is
  rejected with the reason, instead of waiting for ever.
- **Prefill budget.** `max_prefill_tokens` stops several long prompts from landing in the same step.
  A prompt longer than the budget still runs when it is alone in its step.
- **Early stop.** A token equal to `eos_token` ends a request before `max_new_tokens`.
- **Cancel.** `abort(id)` removes a waiting or running request (the client went away) and frees its blocks.

## What the dashboard will read

`scheduler.timeline` has one record per step: which requests prefilled and decoded, which finished,
how many were running and waiting, blocks reserved and blocks in use. That is the data behind the
batching swimlanes. `stats()` gives the live counts, and each `Request` records its wait and
latency in steps.

## Limits

- Time is counted in steps, not seconds. Real step times depend on the batch and are measured
  later, with the load tests.
- One request's prompt is prefilled in a single step (no chunked prefill).
- Reserving the full reply means a request that stops early (end-of-sequence token) held more room
  than it used. The room is returned the moment it finishes.
