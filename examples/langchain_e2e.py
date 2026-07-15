"""End-to-end: build a LangChain agent, then evaluate it with EntroPy.

Run:  python examples/langchain_e2e.py
No API key required - the demo agent is a RunnableLambda.
"""
import random

from langchain_core.runnables import RunnableLambda

from entropy import Suite, Dataset, Case, assert_stable
from entropy.integrations import from_langchain


def _build_agent():
    """A small stochastic LangChain agent (no LLM needed for the demo).

    It echoes the input but occasionally "hallucinates" a wrong answer,
    so EntroPy has something interesting to measure.
    """
    def _fn(x):
        inp = x["input"] if isinstance(x, dict) else x
        r = random.random()
        if r < 0.7:
            return {"output": f"echo: {inp}"}
        if r < 0.9:
            return {"output": "off-topic reply"}
        return {"output": ""}

    return RunnableLambda(_fn)


def main():
    agent = _build_agent()
    wrapped = from_langchain(agent)

    dataset = Dataset([
        Case(input="hello", expected="echo: hello"),
        Case(input="world", expected="echo: world"),
    ])

    results = Suite(seed=42).run(wrapped, dataset, trials=500)

    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # behavioral_entropy measures *action* diversity; the LangChain adapter emits
    # a single constant ActionEvent, so it stays 0 here. Output-level randomness
    # surfaces in behavioral_variance instead.
    assert_stable(
        results,
        success_rate__gt=0.4,
        behavioral_variance__gt=0.0,  # stochastic -> non-zero variance
        drift_score__lt=0.3,
    )
    print("\nassert_stable: OK")


if __name__ == "__main__":
    main()
