"""
Evaluation metrics calculation module.
Computes distance, energy consumption, feasibility rate, runtime, and convergence metrics.
"""
from typing import Dict, List
import numpy as np


class EvaluationMetrics:
    """
    Computes summary metrics across benchmark test runs.
    """

    @staticmethod
    def compute_summary(
        total_distances: List[float],
        feasibility_flags: List[bool],
        runtimes: List[float]
    ) -> Dict[str, float]:
        """
        Calculates aggregate statistics for an algorithm evaluation run.

        Args:
            total_distances (List[float]): List of total route distances across instances/seeds.
            feasibility_flags (List[bool]): Boolean feasibility status per run.
            runtimes (List[float]): Wall-clock execution times (seconds).

        Returns:
            Dict[str, float]: Aggregated metrics dictionary.
        """
        feas_rate = float(np.mean(feasibility_flags)) if feasibility_flags else 0.0
        avg_dist = float(np.mean(total_distances)) if total_distances else float("inf")
        avg_time = float(np.mean(runtimes)) if runtimes else 0.0

        return {
            "feasibility_rate": feas_rate,
            "mean_total_distance": avg_dist,
            "min_total_distance": float(np.min(total_distances)) if total_distances else float("inf"),
            "mean_runtime_seconds": avg_time,
        }
