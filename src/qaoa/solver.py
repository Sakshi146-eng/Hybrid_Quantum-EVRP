"""
QAOA Solver (Legacy Bridge).
Wraps the modular src.quantum.qaoa_solver.QAOAVRPSolver implementation.
"""
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from src.vrp.instance import VRPInstance
from src.quantum.qaoa_solver import QAOAVRPSolver
from src.classical.solution import VRPSolution


class QAOASolver:
    """
    Handles QAOA circuit construction, parameter evaluation, and sampling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary containing p_steps, shots, etc.
        """
        self.config = config or {}
        qaoa_cfg = self.config.get("qaoa", {})
        self.p_steps = qaoa_cfg.get("p_steps", 1)
        self.shots = qaoa_cfg.get("shots", 1024)
        self.max_iter = qaoa_cfg.get("max_iter", 50)
        self.seed = self.config.get("seed", 42)

        self.solver = QAOAVRPSolver(
            p_depth=self.p_steps,
            shots=self.shots,
            max_iter=self.max_iter,
            seed=self.seed,
        )

    def solve(self, instance: VRPInstance) -> VRPSolution:
        """Solves instance using QAOA."""
        return self.solver.solve(instance)

    def sample_best_bitstring(self, counts: Dict[str, int]) -> str:
        """
        Extracts the most frequently measured bitstring from outcome counts.

        Args:
            counts (Dict[str, int]): Dict of bitstring counts.

        Returns:
            str: Most probable bitstring solution.
        """
        if not counts:
            return ""
        return max(counts, key=counts.get)
