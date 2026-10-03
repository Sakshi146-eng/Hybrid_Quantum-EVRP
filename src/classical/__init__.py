"""
Classical VRP Baseline Solvers package.
Provides common interface (BaseVRPSolver) and standardized output (VRPSolution).
"""
from src.classical.solution import VRPSolution
from src.classical.base_solver import BaseVRPSolver, RandomBaselineSolver

__all__ = [
    "VRPSolution",
    "BaseVRPSolver",
    "RandomBaselineSolver",
]
