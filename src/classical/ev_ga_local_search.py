"""
EV-Aware Genetic Algorithm + Local Search Solver.
Combines EV-GA exploration with EV-aware 2-Opt/Relocate polishing and dynamic charging insertion.
"""
import time
from typing import List, Dict, Any, Optional
from src.evrp.instance import EVRPInstance
from src.evrp.cost import calculate_evrp_solution_cost
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution
from src.classical.ev_genetic_algorithm import EVGeneticAlgorithmSolver
from src.classical.ev_local_search import EVLocalSearchSolver


class EVGALocalSearchSolver(BaseVRPSolver):
    """
    EV-Aware GA + Local Search Baseline.
    Provides the standard high-performance classical benchmark for EVRP.
    """

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 50,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.ga_solver = EVGeneticAlgorithmSolver(
            population_size=population_size,
            generations=generations,
            seed=seed,
            **kwargs,
        )
        self.ls_solver = EVLocalSearchSolver(seed=seed)
        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "EV-GA + Local Search"

    def _solve_instance(self, instance: EVRPInstance) -> List[List[int]]:
        t0 = time.perf_counter()

        # Step 1: EV-GA initialization
        ga_sol = self.ga_solver.solve(instance)
        t_ga = time.perf_counter() - t0
        initial_routes = ga_sol.routes
        initial_cost = ga_sol.metadata.get("total_cost", ga_sol.total_distance)

        # Step 2: EV Local Search polish
        t1 = time.perf_counter()
        polished_routes, num_improvements = self.ls_solver.optimize_routes(initial_routes, instance)
        t_ls = time.perf_counter() - t1
        t_total = time.perf_counter() - t0

        cost_eval = calculate_evrp_solution_cost(instance, polished_routes)
        polished_cost = cost_eval["total_cost"]

        # Best-so-far preservation
        if cost_eval["feasible"] and polished_cost <= initial_cost:
            chosen_routes = polished_routes
        else:
            chosen_routes = initial_routes
            cost_eval = ga_sol.metadata.get("evrp_evaluation", cost_eval)

        self.last_metadata = {
            "initial_ga_cost": round(float(initial_cost), 4),
            "polished_cost": round(float(cost_eval["total_cost"]), 4),
            "improvements_count": num_improvements,
            "runtime_breakdown": {
                "ga_seconds": round(t_ga, 6),
                "local_search_seconds": round(t_ls, 6),
                "total_seconds": round(t_total, 6),
            },
            "evrp_evaluation": cost_eval,
            "feasible": cost_eval["feasible"],
        }

        return chosen_routes

    def solve(self, instance: EVRPInstance) -> VRPSolution:
        t0 = time.perf_counter()
        routes = self._solve_instance(instance)
        runtime = time.perf_counter() - t0

        cost_eval = calculate_evrp_solution_cost(instance, routes)

        return VRPSolution(
            algorithm_name=self.algorithm_name,
            routes=routes,
            total_distance=cost_eval["total_distance"],
            feasible=cost_eval["feasible"],
            runtime_seconds=float(runtime),
            metadata={
                **self.last_metadata,
                "evrp_evaluation": cost_eval,
                "total_cost": cost_eval["total_cost"],
                "charging_cost": cost_eval["charging_cost"],
                "total_energy_kwh": cost_eval["total_energy_consumed_kwh"],
                "charging_stops": cost_eval["total_charging_stops"],
            },
        )
