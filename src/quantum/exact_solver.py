"""
Exact Brute-Force Reference Solver for Small Routing Instances.
Exhaustively enumerates all customer permutations to find the globally optimal route.
Used to calculate exact optimality gaps and verify the correctness of QUBO/QAOA solutions.
"""
import itertools
import time
from typing import List, Optional, Dict, Any

from src.vrp.instance import VRPInstance
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution


class ExactBruteForceSolver(BaseVRPSolver):
    """
    Exhaustive brute-force solver that checks all customer permutations.
    Safety limit: only allows problem instances with <= 9 customers (9! = 362,880) to prevent freezing.
    """

    def __init__(self, max_customers: int = 8, **kwargs):
        """
        Args:
            max_customers (int): Maximum customer count allowed for brute-force execution.
        """
        super().__init__(seed=None, **kwargs)
        self.max_customers = max_customers

    @property
    def algorithm_name(self) -> str:
        return "Exact (Brute Force)"

    def _solve_instance(self, instance: VRPInstance) -> List[List[int]]:
        """
        Enumerates all permutations of customer indices and finds minimum distance tour.
        """
        customers = instance.customer_indices
        num_cust = len(customers)

        if num_cust > self.max_customers:
            raise ValueError(
                f"Instance has {num_cust} customers. Exact brute-force is limited to <= {self.max_customers} customers."
            )

        depot = instance.depot_idx

        if num_cust == 0:
            return [[depot, depot]]

        best_route = None
        best_distance = float("inf")
        evaluated_tours = 0

        # Enumerate all permutations
        for perm in itertools.permutations(customers):
            candidate_route = [depot] + list(perm) + [depot]
            dist = instance.calculate_route_distance(candidate_route)
            evaluated_tours += 1
            if dist < best_distance:
                best_distance = dist
                best_route = candidate_route

        self.last_metadata = {
            "evaluated_tours": evaluated_tours,
            "optimal_distance": float(round(best_distance, 4)),
            "num_customers": num_cust,
        }

        return [best_route] if best_route is not None else [[depot, depot]]

    def solve(self, instance: VRPInstance) -> VRPSolution:
        """Solves instance and records exact solution metadata."""
        solution = super().solve(instance)
        if hasattr(self, "last_metadata"):
            solution.metadata.update(self.last_metadata)
        return solution
