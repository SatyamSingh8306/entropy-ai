"""Dataset cases (spec §12): Case and the specialized case types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class Case:
    input: Any
    expected: Any = None
    check: Optional[Callable] = None
    metadata: dict = field(default_factory=dict)

    def is_success(self, output: Any) -> bool:
        if self.check is not None:
            return bool(self.check(output))
        if self.expected is not None:
            return output == self.expected
        return True


@dataclass
class GoldenCase(Case):
    """A known-good input/output pair the agent must reproduce."""

    def is_success(self, output: Any) -> bool:
        return output == self.expected


@dataclass
class Scenario(Case):
    """A stateful scenario; ``setup`` is applied to the environment first."""

    setup: dict = field(default_factory=dict)


@dataclass
class BehaviorCase(Case):
    """A case whose *behavior* (not just output) is checked via ``behavior_check``."""

    behavior_check: Optional[Callable] = None

    def is_behavior_ok(self, run) -> bool:
        return bool(self.behavior_check(run)) if self.behavior_check else True


@dataclass
class FailureCase(Case):
    """A case where the agent is expected to fail gracefully (success == handled)."""

    expect_failure: bool = True

    def is_success(self, output: Any) -> bool:
        # "success" here means the failure was handled (no unhandled error)
        return output is not None


@dataclass
class AdversarialCase(Case):
    """An adversarial input meant to provoke unsafe behavior."""

    attack: str = ""


_CASE_TYPES = {
    "case": Case,
    "golden": GoldenCase,
    "scenario": Scenario,
    "behavior": BehaviorCase,
    "failure": FailureCase,
    "adversarial": AdversarialCase,
}


def make_case(item: dict) -> Case:
    kind = (item.get("type") or "case").lower()
    cls = _CASE_TYPES.get(kind, Case)
    return cls(
        input=item.get("input"),
        expected=item.get("expected"),
        metadata=item.get("metadata", {}),
    )
