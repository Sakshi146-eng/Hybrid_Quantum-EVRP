"""
Classical VRP Baseline Solvers package.
Provides common interface (BaseVRPSolver), standardized output (VRPSolution),
and classical metaheuristics algorithms (Genetic Algorithm).
"""
from src.classical.solution import VRPSolution
from src.classical.base_solver import BaseVRPSolver, RandomBaselineSolver
from src.classical.genetic_algorithm import GeneticAlgorithmSolver, GAConfig

__all__ = [
    "VRPSolution",
    "BaseVRPSolver",
    "RandomBaselineSolver",
    "GeneticAlgorithmSolver",
    "GAConfig",
]
