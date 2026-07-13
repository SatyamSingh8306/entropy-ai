"""Evaluation layers 1-6 (spec §6): group metrics by evaluation concern."""
from __future__ import annotations

from typing import Dict, List

LAYERS: Dict[int, Dict[str, List[str]]] = {
    1: {"name": "Reasoning", "metrics": [
        "plan_quality", "plan_adherence", "reasoning_consistency",
        "planning_stability", "goal_decomposition"]},
    2: {"name": "Tool", "metrics": [
        "tool_correctness", "argument_correctness", "tool_reliability",
        "tool_selection", "tool_thrashing", "hallucinated_tool"]},
    3: {"name": "Execution", "metrics": [
        "task_completion", "step_efficiency", "latency", "cost",
        "retry_rate", "failure_rate", "recovery_score", "cost_stability"]},
    4: {"name": "Behavioral", "metrics": [
        "behavioral_entropy", "behavioral_variance", "trajectory_diversity",
        "goal_stability", "loop_detection", "exploration_efficiency_metric",
        "emergent_behavior"]},
    5: {"name": "Robustness", "metrics": [
        "robustness", "robustness_entropy", "recovery_score", "drift_score",
        "reliability_score"]},
    6: {"name": "Drift", "metrics": [
        "drift_score", "trajectory_drift", "tool_usage_drift",
        "cost_drift", "behavior_fingerprint"]},
}


def layer_metrics(layer: int) -> List[str]:
    return list(LAYERS[layer]["metrics"])


def layer_name(layer: int) -> str:
    return LAYERS[layer]["name"]
