"""Run locally: python examples/evaluation_comparison.py (no API key needed)."""

from entropy import compare_agents, evaluate_detailed

cases = {" Hello ": "hello", "WORLD": "world", " Test ": "test"}


def baseline(text):
    return text.strip()


def candidate(text):
    return text.strip().lower()


report = evaluate_detailed(candidate, cases, trials=10, pass_k=[1, 5, 10])
print(report["pass_at_k"])
print(report["cases"][0]["confidence_interval"])

comparison = compare_agents(
    {"baseline": baseline, "candidate": candidate},
    cases,
    trials=10,
    resamples=1000,
    seed=42,
)
print(comparison["comparisons"]["candidate"]["status"])
