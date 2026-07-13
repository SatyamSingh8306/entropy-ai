"""Metric registry, Batch container, and shared math helpers."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

# --------------------------------------------------------------------------
# Batch: everything a metric needs, computed by Suite from one input's trials
# --------------------------------------------------------------------------
@dataclass
class Batch:
    inp: Any
    outputs: List[Any] = field(default_factory=list)
    events: List[list] = field(default_factory=list)        # list-of-Events per trial
    behaviors: List[List[str]] = field(default_factory=list)  # trajectory signature per trial
    actions: List[str] = field(default_factory=list)          # primary action label per trial
    tool_calls: List[List[str]] = field(default_factory=list) # tool names per trial
    successes: List[bool] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    errored: List[bool] = field(default_factory=list)
    recovered: List[bool] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Metric base + registry (plugin system: @entropy.metric)
# --------------------------------------------------------------------------
class Metric:
    def compute(self, b: Batch) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


_REGISTRY: dict[str, type] = {}
_ORDER: List[str] = []
_DEFAULT: List[str] = []


def metric(name: str | None = None) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        key = name or cls.__name__
        _REGISTRY[key] = cls
        if key not in _ORDER:
            _ORDER.append(key)
        return cls
    return deco


def default(name: str) -> None:
    """Mark a registered metric as part of the canonical Suite output."""
    if name not in _DEFAULT:
        _DEFAULT.append(name)


def get_metric(name: str) -> Metric:
    return _REGISTRY[name]()


def default_metrics() -> List[str]:
    return list(_DEFAULT)


# --------------------------------------------------------------------------
# Math helpers
# --------------------------------------------------------------------------
def shannon(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def norm_entropy(sigs: List[Any]) -> float:
    """Normalized Shannon entropy of a signature distribution, in [0, 1]."""
    if not sigs:
        return 0.0
    counts = list(Counter(sigs).values())
    n = len(set(sigs))
    if n <= 1:
        return 0.0
    return shannon(counts) / math.log(n)


def bernoulli_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def variance(xs: List[float]) -> float:
    return statistics.variance(xs) if len(xs) > 1 else 0.0


def flat_tool_calls(b: Batch) -> List[str]:
    out: List[str] = []
    for trial in b.tool_calls:
        out.extend(trial)
    return out


def event_names(b: Batch, etype: str) -> List[List[str]]:
    """Event names of a given type, per trial."""
    names: List[List[str]] = []
    for trial in b.events:
        names.append([e.name for e in trial if e.type == etype])
    return names


def half_drift(sigs: List[Any]) -> float:
    """Total-variation drift between first/second half of the trial sequence."""
    n = len(sigs)
    if n < 4:
        return 0.0
    half = n // 2
    d1 = Counter(sigs[:half])
    d2 = Counter(sigs[half:])
    t1, t2 = sum(d1.values()), sum(d2.values())
    tv = 0.0
    for k in set(d1) | set(d2):
        p1 = d1.get(k, 0) / t1 if t1 else 0.0
        p2 = d2.get(k, 0) / t2 if t2 else 0.0
        tv += abs(p1 - p2)
    return tv / 2.0


# --- distance / divergence helpers (vision doc § metrics 4, 18) ----------
def levenshtein(a: List[Any], b: List[Any]) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def avg_pairwise_lev(seqs: List[List[Any]]) -> float:
    """Mean normalized Levenshtein distance across all trajectory pairs."""
    n = len(seqs)
    if n < 2:
        return 0.0
    maxlen = max((len(s) for s in seqs), default=1) or 1
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += levenshtein(seqs[i], seqs[j]) / maxlen
            pairs += 1
    return total / pairs if pairs else 0.0


def _dist(counts: List[float]) -> List[float]:
    total = sum(counts) or 1.0
    return [c / total for c in counts]


def js_divergence(c1: List[float], c2: List[float]) -> float:
    """Jensen-Shannon divergence (base 2), normalized to [0, 1]."""
    p, q = _dist(c1), _dist(c2)
    m = [0.5 * (a + b) for a, b in zip(p, q)]

    def _kl(a, b):
        s = 0.0
        for x, y in zip(a, b):
            if x > 0 and y > 0:
                s += x * math.log(x / y)
        return s

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return min(1.0, js / math.log(2)) if js > 0 else 0.0
