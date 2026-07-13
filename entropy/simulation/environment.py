"""Environment simulator (spec §11): stateful env with rewards."""
from __future__ import annotations

from typing import Any, Dict, Tuple, Protocol


class Environment(Protocol):
    def step(self, action: Any) -> Tuple[Dict, float, bool, Dict]: ...
    def reset(self) -> None: ...


class StatefulEnv:
    """Generic key/value environment. Actions that are dicts update state.

    Reward/done are read from ``state['reward']`` / ``state['done']``.
    """

    def __init__(self, state: Dict[str, Any] | None = None):
        self.state = dict(state or {})

    def reset(self) -> None:
        self.state = {}

    def step(self, action: Any) -> Tuple[Dict, float, bool, Dict]:
        if isinstance(action, dict):
            self.state.update(action)
        reward = float(self.state.get("reward", 0.0))
        done = bool(self.state.get("done", False))
        return dict(self.state), reward, done, {}


class GridWorld(StatefulEnv):
    """A tiny 2D grid world for reward/drift experiments."""

    def __init__(self, width: int = 5, height: int = 5, goal=(4, 4)):
        super().__init__({"x": 0, "y": 0, "reward": 0.0, "done": False})
        self.width, self.height, self.goal = width, height, goal

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        x, y = self.state["x"], self.state["y"]
        if action == "up":
            y = max(0, y - 1)
        elif action == "down":
            y = min(self.height - 1, y + 1)
        elif action == "left":
            x = max(0, x - 1)
        elif action == "right":
            x = min(self.width - 1, x + 1)
        self.state["x"], self.state["y"] = x, y
        self.state["done"] = (x, y) == self.goal
        self.state["reward"] = 1.0 if self.state["done"] else -0.01
        return dict(self.state), self.state["reward"], self.state["done"], {}
