"""End-to-end demo: evaluate a stochastic agent with EntroPy."""
import random

from entropy import Suite, Dataset, Case, assert_stable


def fickle_agent(inp):
    # A non-deterministic agent: mostly returns "A", sometimes "B"/"C".
    r = random.random()
    if r < 0.6:
        return "A"
    if r < 0.85:
        return "B"
    return "C"


def main():
    dataset = Dataset([Case(input="q1", expected="A"), Case(input="q2", expected="B")])
    suite = Suite(seed=42)
    results = suite.run(fickle_agent, dataset, trials=300)

    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # This stochastic agent is non-deterministic but only ~43% correct on
    # these two cases; assert the things that should hold.
    assert_stable(
        results,
        success_rate__gt=0.3,
        behavioral_entropy__gt=0.0,   # non-deterministic -> non-zero entropy
        drift_score__lt=0.2,
    )
    print("\nassert_stable: OK")


if __name__ == "__main__":
    main()
