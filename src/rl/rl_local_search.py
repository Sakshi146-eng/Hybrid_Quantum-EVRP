"""
RL + Classical Local Search Baseline Solver.
Uses the trained/initialized EVRLAgent to generate initial routes, then applies
EV-aware 2-Opt and customer relocation local search with dynamic charging insertion.
"""
import time
from typing import List, Dict, Any, Optional
from src.evrp.instance import EVRPInstance
from src.evrp.cost import calculate_evrp_solution_cost
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution
from src.rl.agent import EVRLAgent
from src.classical.ev_local_search import EVLocalSearchSolver


class RLLocalSearchSolver(BaseVRPSolver):
    """
    RL + Local Search Baseline.
    Evaluates whether classical neighborhood polishing enhances RL-constructed EV routes.
    """

    def __init__(
        self,
        rl_agent: Optional[EVRLAgent] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.rl_agent = rl_agent or EVRLAgent(seed=seed)
        self.ls_solver = EVLocalSearchSolver(seed=seed)
        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "RL + Classical Local Search"

    def _solve_instance(self, instance: EVRPInstance) -> List[List[int]]:
        t0 = time.perf_counter()

        # 1. RL route generation
        rl_sol = self.rl_agent.solve(instance)
        initial_routes = rl_sol.routes
        initial_cost = rl_sol.metadata.get("total_cost", rl_sol.total_distance)
        t_rl = time.perf_counter() - t0

        # 2. Local search polishing
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
            cost_eval = rl_sol.metadata.get("evrp_evaluation", cost_eval)

        self.last_metadata = {
            "initial_rl_cost": round(float(initial_cost), 4),
            "polished_cost": round(float(cost_eval["total_cost"]), 4),
            "improvements_count": num_improvements,
            "runtime_breakdown": {
                "rl_seconds": round(t_rl, 6),
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
