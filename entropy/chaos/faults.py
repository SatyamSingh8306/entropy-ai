"""Fault injection (spec §10): 8 fault types as agent wrappers."""
from __future__ import annotations

import random
from typing import Callable, Dict, List

from ..core.models import AgentRun, ErrorEvent, ToolCallEvent, MemoryWriteEvent


def _wrap(agent: Callable, fault_fn, p: float, rng: random.Random) -> Callable:
    def wrapped(inp):
        if rng.random() < p:
            return fault_fn(agent, inp)
        return agent(inp)
    wrapped.__name__ = getattr(fault_fn, "__name__", "fault")
    return wrapped


# --- individual faults ----------------------------------------------------
def api_fail(agent, inp):
    raise RuntimeError("CHAOS: simulated API failure")


def timeout(agent, inp):
    raise TimeoutError("CHAOS: simulated timeout")


def rate_limit(agent, inp):
    raise RuntimeError("CHAOS: 429 rate limited")


def network_delay(agent, inp):
    raise ConnectionError("CHAOS: simulated network delay/reset")


def tool_crash(agent, inp):
    res = agent(inp)
    if isinstance(res, AgentRun):
        events = [ToolCallEvent("__crashed__") if e.type == "tool" else e
                  for e in res.events]
        return AgentRun(run_id=res.run_id, input=inp, output=res.output,
                        events=events, cost=res.cost,
                        metadata={**res.metadata, "error": "tool_crash"})
    return AgentRun(run_id="", input=inp, output=res,
                    events=[ToolCallEvent("__crashed__")], metadata={"error": "tool_crash"})


def memory_corrupt(agent, inp):
    res = agent(inp)
    if isinstance(res, AgentRun):
        events = [MemoryWriteEvent(e.name, {**e.data, "polluting": True})
                  if e.type == "memory_write" else e for e in res.events]
        return AgentRun(run_id=res.run_id, input=inp, output=res.output,
                        events=events, cost=res.cost, metadata=res.metadata)
    return res


def token_limit(agent, inp, limit: int = 16):
    res = agent(inp)
    out = res.output if isinstance(res, AgentRun) else res
    text = out if isinstance(out, str) else str(out)
    if len(text) > limit:
        truncated = text[:limit]
        if isinstance(res, AgentRun):
            return AgentRun(run_id=res.run_id, input=inp, output=truncated,
                            events=res.events, cost=res.cost,
                            metadata={**res.metadata, "error": "token_limit"})
        return AgentRun(run_id="", input=inp, output=truncated, metadata={"error": "token_limit"})
    return res


def malformed_output(agent, inp):
    res = agent(inp)
    if isinstance(res, AgentRun):
        return AgentRun(run_id=res.run_id, input=inp, output="[[MALFORMED]]",
                        events=res.events, cost=res.cost,
                        metadata={**res.metadata, "error": "malformed"})
    return AgentRun(run_id="", input=inp, output="[[MALFORMED]]", metadata={"error": "malformed"})


FAULTS: Dict[str, Callable] = {
    "api_fail": api_fail,
    "timeout": timeout,
    "rate_limit": rate_limit,
    "network_delay": network_delay,
    "tool_crash": tool_crash,
    "memory_corrupt": memory_corrupt,
    "token_limit": token_limit,
    "malformed_output": malformed_output,
}


def inject(agent: Callable, fault: str, p: float = 1.0, seed: int = 0) -> Callable:
    """Wrap ``agent`` so fault ``fault`` fires with probability ``p``."""
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}; have {sorted(FAULTS)}")
    return _wrap(agent, FAULTS[fault], p, random.Random(seed))


def available() -> List[str]:
    return list(FAULTS)
