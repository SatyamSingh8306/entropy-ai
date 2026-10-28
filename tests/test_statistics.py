"""Known-value and boundary tests for statistical evaluation methods."""

import math
import random

import pytest
from entropy import Suite, bootstrap_interval, pass_at_k, wilson_interval


def test_pass_at_k_matches_exact_subset_enumeration():
    for total in range(1, 9):
        for correct in range(total + 1):
            for k in range(1, total + 1):
                expected = 1 - math.comb(total - correct, k) / math.comb(total, k)
                assert pass_at_k(total, correct, k) == pytest.approx(expected)


def test_pass_at_k_retains_small_probabilities():
    assert pass_at_k(10**16, 1) == pytest.approx(1e-16, abs=1e-30)
    assert pass_at_k(10**18, 10**18 - 1) == 1.0


@pytest.mark.parametrize(
    "args",
    [
        (0, 0, 1),
        (3, 4, 1),
        (3, -1, 1),
        (3, 1, 0),
        (3, 1, 4),
        (True, 0, 1),
        (3, True, 1),
        (3, 1, 1.5),
    ],
)
def test_invalid_pass_at_k(args):
    with pytest.raises(ValueError):
        pass_at_k(*args)


def test_wilson_known_value_and_extremes():
    assert wilson_interval(5, 10) == pytest.approx([0.2365930905, 0.7634069095])
    assert wilson_interval(10, 10) == pytest.approx([0.7224672001, 1])
    assert wilson_interval(0, 10) == pytest.approx([0, 0.2775327999])
    assert wilson_interval(0, 0) == [0, 1]
    assert wilson_interval(0, 10, math.nextafter(1, 0))[0] == 0
    narrow = wilson_interval(5, 10, 0.8)
    wide = wilson_interval(5, 10, 0.99)
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


@pytest.mark.parametrize(
    "correct,total", [(-1, 1), (2, 1), (0, -1), (True, 1), (0, 1.5)]
)
def test_invalid_wilson_counts(correct, total):
    with pytest.raises(ValueError):
        wilson_interval(correct, total)


@pytest.mark.parametrize(
    "confidence", [0, 1, -0.5, float("nan"), float("inf"), True, "95"]
)
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        wilson_interval(1, 2, confidence)
    with pytest.raises(ValueError):
        bootstrap_interval([1, 2], confidence=confidence)


def test_bootstrap_reproducible_and_does_not_mutate_global_rng():
    random.seed(23)
    state = random.getstate()
    first = bootstrap_interval(iter([1, 2, 3, 4]), seed=7, resamples=500)
    assert first == bootstrap_interval([1, 2, 3, 4], seed=7, resamples=500)
    assert 1 <= first[0] < 2.5 < first[1] <= 4
    assert random.getstate() == state
    assert bootstrap_interval([3, 3, 3]) == [3, 3]


@pytest.mark.parametrize(
    "values", [[], [1], [1, float("nan")], [1, float("inf")], [1, "2"], [1, True]]
)
def test_invalid_bootstrap_values(values):
    with pytest.raises(ValueError):
        bootstrap_interval(values)


@pytest.mark.parametrize("resamples", [0, 1, -1, True, 2.5])
def test_invalid_resamples(resamples):
    with pytest.raises(ValueError):
        bootstrap_interval([1, 2], resamples=resamples)


def test_suite_confidence_interval_uses_all_trials_not_average_bounds():
    suite = Suite(metrics=["success_rate", "confidence_interval"])
    result = suite.run(lambda inp: inp, {"yes": "yes", "no": "yes"}, trials=10)
    assert result["success_rate"] == 0.5
    assert result["confidence_interval"] == wilson_interval(10, 20)
    assert suite.run(lambda inp: inp, [], trials=1)["confidence_interval"] == [0, 1]


def test_sync_evaluation_rejects_unawaited_coroutines():
    async def agent(inp):
        return inp

    with pytest.raises(TypeError, match="evaluate_async"):
        Suite().run(agent, ["a"], trials=1)


def test_bootstrap_large_finite_values_do_not_overflow():
    assert bootstrap_interval([1e308, 1e308], resamples=10) == [1e308, 1e308]
    bounds = bootstrap_interval([-1e308, 1e308], resamples=2, seed=5)
    assert all(math.isfinite(value) for value in bounds)
