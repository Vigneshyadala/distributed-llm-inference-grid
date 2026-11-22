"""Continuous batching: decide, at every step, which requests run together.

A model step costs about the same whether it serves one sequence or eight, so serving many at
once is nearly free. The question is how to form the batch. STATIC batching takes a group, runs
it until its longest member is done, and only then takes the next group: short requests sit idle
and new ones wait. CONTINUOUS batching (as in vLLM and Orca) re-forms the batch at every step:
a finished request leaves at once and a waiting one takes its place.

    static      A A A A A A A A      continuous   A A A A A A A A
                B B B . . . . .                   B B B C C C C C
                C C C C C C . .                   (C started the step B left)

Every step is one of two kinds of work per request:

    prefill  the whole prompt goes through the model at once; this also produces the first token
    decode   one token goes in, the next one comes out (the keys and values of the earlier
             tokens come from the paged KV cache)

The scheduler does not run any model. `next_step()` says who prefills and who decodes, the caller
(the master) runs the step and reports the tokens with `complete_step()`. That keeps the policy
testable without torch and lets the same scheduler drive fake models, one worker or the whole grid.

Admission policy, chosen to be simple and safe:

    first come, first served   a request is never overtaken, so nobody starves
    reserve, do not guess      a request is admitted only if the cache can hold its prompt AND all
                               tokens it may still generate, counting what running requests have
                               reserved. A running request therefore can never hit `OutOfBlocks`
                               in the middle of a reply, and no request needs to be thrown out or
                               recomputed. The price is that room is held a little longer than
                               strictly needed.
"""
from collections import deque

WAITING, RUNNING, FINISHED, REJECTED, ABORTED = "waiting", "running", "finished", "rejected", "aborted"


class SchedulerError(RuntimeError):
    """The scheduler was used incorrectly (duplicate id, wrong step order, bad tokens, ...)."""


class Request:
    """One generation request: `prompt_len` tokens in, at most `max_new_tokens` tokens out."""

    def __init__(self, request_id, prompt_len, max_new_tokens):
        for name, value in (("prompt_len", prompt_len), ("max_new_tokens", max_new_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive whole number, not {value!r}")
        self.id = request_id
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.state = WAITING
        self.generated = 0
        self.reason = None            # why it was rejected or aborted, or "eos" / "length" when finished
        self.arrival_step = None      # step number when it was submitted
        self.admitted_step = None     # step in which its prefill ran
        self.finished_step = None     # step in which its last token was produced

    @property
    def total_tokens(self):
        """Most tokens it can ever need room for in the cache."""
        return self.prompt_len + self.max_new_tokens

    @property
    def wait_steps(self):
        """Steps spent waiting before the prefill (None until admitted)."""
        return None if self.admitted_step is None else self.admitted_step - self.arrival_step

    @property
    def latency_steps(self):
        """Steps from submission until the last token (None until finished)."""
        return None if self.finished_step is None else self.finished_step - self.arrival_step + 1

    def __repr__(self):
        return f"Request({self.id!r}, prompt={self.prompt_len}, new={self.max_new_tokens}, {self.state})"


class StepPlan:
    """What one step does: `prefill` requests process their prompts, `decode` requests add a token."""

    def __init__(self, step, prefill, decode):
        self.step, self.prefill, self.decode = step, list(prefill), list(decode)

    @property
    def requests(self):
        return self.prefill + self.decode

    @property
    def tokens_in(self):
        """Tokens that go through the model this step."""
        return sum(r.prompt_len for r in self.prefill) + len(self.decode)

    def __repr__(self):
        return (f"StepPlan(step={self.step}, prefill={[r.id for r in self.prefill]}, "
                f"decode={[r.id for r in self.decode]})")


class Scheduler:
    """Admits waiting requests into the running batch and retires finished ones.

    cache               a `PagedKVCache` (only its block bookkeeping is used)
    max_batch_size      most requests running at once
    max_prefill_tokens  most prompt tokens to prefill in one step (None = no limit). A long
                        prompt is still admitted if it is the only one in its step, so it cannot
                        be starved; the limit only stops several long prompts from landing at once.
    eos_token           a generated token equal to this ends the request early
    """

    def __init__(self, cache, max_batch_size=8, max_prefill_tokens=None, eos_token=None):
        if not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError(f"max_batch_size must be a positive whole number, not {max_batch_size!r}")
        if max_prefill_tokens is not None and (not isinstance(max_prefill_tokens, int) or max_prefill_tokens < 1):
            raise ValueError(f"max_prefill_tokens must be a positive whole number, not {max_prefill_tokens!r}")
        self.cache = cache
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.eos_token = eos_token
        self.step_number = 0
        self._waiting = deque()
        self._running = []
        self._requests = {}
        self._pending = None
        self.finished = []
        self.rejected = []
        self.aborted = []
        self.timeline = []   # one record per completed step, for the dashboard's batching view

    # -------------------------------------------------------------------- requests

    def submit(self, request):
        """Queue a request. One that could never fit, even in an empty cache, is rejected at once."""
        if request.id in self._requests:
            raise SchedulerError(f"request {request.id!r} was already submitted")
        if request.state != WAITING:
            raise SchedulerError(f"request {request.id!r} is not new (state {request.state})")
        self._requests[request.id] = request
        request.arrival_step = self.step_number
        need = self._blocks(request)
        if need > self.cache.num_blocks:
            request.state = REJECTED
            request.reason = (f"needs {need} blocks for {request.total_tokens} tokens, "
                              f"the whole cache has {self.cache.num_blocks}")
            self.rejected.append(request)
        else:
            self._waiting.append(request)
        return request

    def abort(self, request_id, reason="aborted"):
        """Cancel a waiting or running request (for example, the client went away). Between steps only."""
        self._no_step_pending("abort")
        request = self._requests.get(request_id)
        if request is None or request.state not in (WAITING, RUNNING):
            return False
        if request.state == WAITING:
            self._waiting.remove(request)
        else:
            self._running.remove(request)
            self.cache.free_sequence(request.id)
        request.state, request.reason = ABORTED, reason
        self.aborted.append(request)
        return True

    def get(self, request_id):
        try:
            return self._requests[request_id]
        except KeyError:
            raise SchedulerError(f"unknown request {request_id!r}") from None

    # ----------------------------------------------------------------------- steps

    @property
    def waiting(self):
        return list(self._waiting)

    @property
    def running(self):
        return list(self._running)

    @property
    def idle(self):
        return not self._waiting and not self._running

    @property
    def reserved_blocks(self):
        """Blocks promised to running requests (what they hold now plus what they may still need)."""
        return sum(self._blocks(r) for r in self._running)

    def next_step(self):
        """Plan the next step, admitting waiting requests that fit. None when there is no work."""
        self._no_step_pending("plan another step")
        decode = list(self._running)
        prefill = self._admit()
        if not prefill and not decode:
            return None
        self._pending = StepPlan(self.step_number, prefill, decode)
        return self._pending

    def complete_step(self, plan, tokens):
        """Report the step's results: `tokens` maps each request id in the plan to its new token.

        Returns the requests that finished in this step. Their cache blocks are already freed.
        """
        if plan is not self._pending:
            raise SchedulerError("complete_step needs the plan that next_step returned last")
        expected = {r.id for r in plan.requests}
        if set(tokens) != expected:
            missing, extra = expected - set(tokens), set(tokens) - expected
            raise SchedulerError(f"tokens must cover exactly the planned requests "
                                 f"(missing {sorted(map(str, missing))}, unexpected {sorted(map(str, extra))})")
        done = []
        for request in plan.requests:
            request.generated += 1
            if self.eos_token is not None and tokens[request.id] == self.eos_token:
                self._finish(request, plan.step, "eos")
                done.append(request)
            elif request.generated >= request.max_new_tokens:
                self._finish(request, plan.step, "length")
                done.append(request)
        self.timeline.append({
            "step": plan.step,
            "prefill": [r.id for r in plan.prefill], "decode": [r.id for r in plan.decode],
            "finished": [r.id for r in done],
            "running": len(self._running), "waiting": len(self._waiting),
            "reserved_blocks": self.reserved_blocks, "blocks_used": self.cache.num_blocks - self.cache.free_blocks,
            "tokens_in": plan.tokens_in,
        })
        self._pending = None
        self.step_number += 1
        return done

    def stats(self):
        """Numbers for the dashboard."""
        return {
            "step": self.step_number, "waiting": len(self._waiting), "running": len(self._running),
            "finished": len(self.finished), "rejected": len(self.rejected), "aborted": len(self.aborted),
            "max_batch_size": self.max_batch_size, "reserved_blocks": self.reserved_blocks,
            "blocks_total": self.cache.num_blocks, "blocks_free": self.cache.free_blocks,
        }

    # ------------------------------------------------------------------- internals

    def _blocks(self, request):
        return self.cache.blocks_for_tokens(request.total_tokens)

    def _admit(self):
        admitted, reserved, prompt_tokens = [], self.reserved_blocks, 0
        while self._waiting and len(self._running) < self.max_batch_size:
            request = self._waiting[0]
            need = self._blocks(request)
            if reserved + need > self.cache.num_blocks:
                break                                      # first come, first served: do not skip it
            if (self.max_prefill_tokens is not None and admitted
                    and prompt_tokens + request.prompt_len > self.max_prefill_tokens):
                break
            if not self.cache.can_fit(request.prompt_len):  # cannot happen while reservations hold
                break
            self._waiting.popleft()
            self.cache.add_sequence(request.id)
            request.state, request.admitted_step = RUNNING, self.step_number
            self._running.append(request)
            admitted.append(request)
            reserved += need
            prompt_tokens += request.prompt_len
        return admitted

    def _finish(self, request, step, reason):
        self._running.remove(request)
        self.cache.free_sequence(request.id)
        request.state, request.reason, request.finished_step = FINISHED, reason, step
        self.finished.append(request)

    def _no_step_pending(self, what):
        if self._pending is not None:
            raise SchedulerError(f"cannot {what}: step {self._pending.step} has not been completed")


def run_to_completion(scheduler, execute, arrivals=None, max_steps=100000):
    """Drive a scheduler until everything is done. `execute(plan)` runs a step, returns {id: token}.

    `arrivals` maps a step number to the requests that arrive at the start of that step.
    The master uses the same loop with a real model behind `execute`.
    """
    arrivals = dict(arrivals or {})
    last_arrival = max(arrivals, default=-1)
    while scheduler.step_number < max_steps:
        for request in arrivals.pop(scheduler.step_number, []):
            scheduler.submit(request)
        plan = scheduler.next_step()
        if plan is None:
            if scheduler.step_number >= last_arrival:
                return scheduler
            scheduler.step_number += 1   # idle step: nothing to run, waiting for an arrival
            continue
        scheduler.complete_step(plan, execute(plan))
    raise SchedulerError(f"still busy after {max_steps} steps")
