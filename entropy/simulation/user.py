"""User simulators (spec §11): scripted, random, and optional LLM-backed."""
from __future__ import annotations

import random
from typing import Iterable, List, Optional, Protocol


class UserSimulator(Protocol):
    def next(self) -> str: ...
    def reset(self) -> None: ...


class ScriptedUserSimulator:
    """Replays a fixed conversation script, looping at the end."""

    def __init__(self, script: Iterable[str]):
        self.script = list(script)
        self.i = 0

    def next(self) -> str:
        if not self.script:
            return ""
        msg = self.script[self.i % len(self.script)]
        self.i += 1
        return msg

    def reset(self) -> None:
        self.i = 0


class RandomUserSimulator:
    """Samples user turns from a pool."""

    def __init__(self, pool: Iterable[str], seed: int = 0):
        self.pool = list(pool)
        self.seed = seed
        self.rng = random.Random(seed)

    def next(self) -> str:
        return self.rng.choice(self.pool) if self.pool else ""

    def reset(self) -> None:
        self.rng = random.Random(self.seed)


class LLMUserSimulator:
    """LLM-driven user; requires an OpenAI-compatible client (lazy import)."""

    def __init__(self, client=None, model: str = "gpt-4o",
                 system: str = "You are a user testing an assistant."):
        self.client = client
        self.model = model
        self.system = system
        self.history: List[dict] = []

    def next(self, last_agent_msg: str = "") -> str:
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception as e:  # pragma: no cover
                raise RuntimeError("LLMUserSimulator needs an OpenAI client") from e
        self.history.append({"role": "assistant", "content": last_agent_msg})
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": self.system},
                                         *self.history])
        msg = resp.choices[0].message.content
        self.history.append({"role": "user", "content": msg})
        return msg

    def reset(self) -> None:
        self.history = []
