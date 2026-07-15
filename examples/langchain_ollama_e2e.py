"""End-to-end: build a LangChain agent with create_agent + Ollama, then evaluate.

Model: minimax-m3:cloud via langchain-ollama (ChatOllama).
Requires a running Ollama server with `ollama pull minimax-m3:cloud`.

Run:  python examples/langchain_ollama_e2e.py
"""
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from entropy import Suite, Dataset, Case, assert_stable
from entropy.integrations import from_langgraph


@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for a city."""
    return f"It is sunny in {city}."


def _build_agent():
    # Low temperature + small context keeps minimax-m3:cloud stable on a 4 GB GPU.
    llm = ChatOllama(model="minimax-m3:cloud", temperature=0.0,
                    num_ctx=2048, num_predict=512)
    return create_agent(llm, tools=[get_weather],
                        system_prompt="You are a helpful assistant. Answer concisely.")


def main():
    agent = _build_agent()
    base = from_langgraph(agent)  # create_agent -> CompiledStateGraph

    # minimax-m3:cloud can 500 intermittently on small GPUs; retry so transient
    # errors don't sink the whole evaluation.
    import time

    def robust(inp):
        last = None
        for attempt in range(3):
            try:
                return base(inp)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.0 * (attempt + 1))
        raise last

    wrapped = robust

    # Pre-flight: make sure Ollama is reachable before running trials blind.
    # The model can 500 intermittently on small GPUs, so retry a few times.
    last_err = None
    for _ in range(3):
        try:
            wrapped("ping")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    else:
        raise SystemExit(
            "Could not reach Ollama / model. Start it with `ollama serve` "
            "and run `ollama pull minimax-m3:cloud`.\n"
            f"Underlying error: {last_err}"
        )

    dataset = Dataset([
        Case(input="What is the weather in Paris?",
             check=lambda o: "sunny" in (o or "").lower()),
        Case(input="Tell me a joke.",
             check=lambda o: bool(o) and len(o) > 15),
    ])

    results = Suite(seed=42).run(wrapped, dataset, trials=3)

    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # NOTE: behavioral_entropy (action diversity) stays 0 because the adapter
    # emits a single constant event; output-level variation shows in variance.
    assert_stable(
        results,
        success_rate__gt=0.5,
        drift_score__lt=0.6,
    )
    print("\nassert_stable: OK")


if __name__ == "__main__":
    main()
