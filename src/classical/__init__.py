"""
Classical VRP and EVRP Baseline Solvers package.
Provides common interface (BaseVRPSolver), standardized output (VRPSolution),
and classical metaheuristics algorithms (Genetic Algorithm, Local Search, EV-GA, EV-GA+LS).
"""
from src.classical.solution import VRPSolution
from src.classical.base_solver import BaseVRPSolver, RandomBaselineSolver
from src.classical.genetic_algorithm import GeneticAlgorithmSolver, GAConfig
from src.classical.local_search import (
    two_opt_single_route,
    relocate_single_route,
    swap_single_route,
    apply_comprehensive_local_search,
)
from src.classical.ga_local_search import GALocalSearchSolver
from src.classical.ev_genetic_algorithm import EVGeneticAlgorithmSolver
from src.classical.ev_local_search import EVLocalSearchSolver, apply_ev_local_search_to_route
from src.classical.ev_ga_local_search import EVGALocalSearchSolver

__all__ = [
    "VRPSolution",
    "BaseVRPSolver",
    "RandomBaselineSolver",
    "GeneticAlgorithmSolver",
    "GAConfig",
    "two_opt_single_route",
    "relocate_single_route",
    "swap_single_route",
    "apply_comprehensive_local_search",
    "GALocalSearchSolver",
    "EVGeneticAlgorithmSolver",
    "EVLocalSearchSolver",
    "apply_ev_local_search_to_route",
    "EVGALocalSearchSolver",
]
