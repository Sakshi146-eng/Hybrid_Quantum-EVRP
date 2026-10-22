"""
Genetic Algorithm with Local Search (GA + Local Search) Solver.
Provides a strong, fair classical baseline that combines genetic evolution with
2-Opt, Relocate, and Swap local neighborhood improvements.
"""
import time
from typing import List, Dict, Any, Optional

from src.vrp.instance import VRPInstance
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.classical.local_search import apply_comprehensive_local_search


class GALocalSearchSolver(BaseVRPSolver):
    """
    Classical GA coupled with 2-Opt and local neighborhood refinement.
    Ensures that hybrid quantum-classical methods are compared against an optimized
    classical baseline rather than an unpolished raw heuristic.
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.25,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.ga_solver = GeneticAlgorithmSolver(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            seed=seed,
            **kwargs,
        )
        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "GA + Local Search"

    def _solve_instance(self, instance: VRPInstance) -> List[List[int]]:
        t0 = time.perf_counter()

        # Step 1: Classical GA execution
        ga_sol: VRPSolution = self.ga_solver.solve(instance)
        t_ga = time.perf_counter() - t0
        initial_routes = ga_sol.routes
        initial_distance = ga_sol.total_distance

        # Step 2: Comprehensive Local Search
        t1 = time.perf_counter()
        polished_routes, polished_distance, num_improvements = apply_comprehensive_local_search(
            initial_routes, instance
        )
        t_ls = time.perf_counter() - t1
        t_total = time.perf_counter() - t0

        self.last_metadata = {
            "local_search_applied": True,
            "initial_ga_distance": round(float(initial_distance), 4),
            "polished_distance": round(float(polished_distance), 4),
            "improvements_count": num_improvements,
            "runtime_breakdown": {
                "ga_runtime_seconds": round(t_ga, 6),
                "local_search_runtime_seconds": round(t_ls, 6),
                "total_runtime_seconds": round(t_total, 6),
            },
        }

        # Best-so-far preservation
        if polished_distance <= initial_distance:
            return polished_routes
        return initial_routes

    def solve(self, instance: VRPInstance) -> VRPSolution:
        solution = super().solve(instance)
        if hasattr(self, "last_metadata"):
            solution.metadata.update(self.last_metadata)
        return solution
