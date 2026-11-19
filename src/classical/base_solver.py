"""
Abstract Base Class for Classical VRP Solvers.
Defines the standard solver interface, time measurement, cost calculation,
payload load computation, and feasibility validation.
"""
from abc import ABC, abstractmethod
import time
from typing import List, Optional, Dict, Any
import numpy as np

from src.vrp.instance import VRPInstance
from src.classical.solution import VRPSolution


class BaseVRPSolver(ABC):
    """
    Abstract base class for all classical VRP algorithms (GA, ACO, SA, Tabu Search).
    Enforces common input/output interfaces across algorithms.
    """

    def __init__(self, seed: Optional[int] = 42, **kwargs):
        """
        Args:
            seed (Optional[int]): Seed for random number generators to ensure reproducibility.
            **kwargs: Additional algorithm-specific hyperparameters.
        """
        self.seed = seed
        self.params = kwargs

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Returns human-readable name of the classical algorithm."""
        pass

    @abstractmethod
    def _solve_instance(self, instance: VRPInstance) -> List[List[int]]:
        """
        Subclass solver logic.
        Must return a list of vehicle routes, where each route is a list of node indices
        starting and ending at instance.depot_idx.

        Args:
            instance (VRPInstance): Classical VRP problem instance.

        Returns:
            List[List[int]]: Vehicle routes.
        """
        pass

    def solve(self, instance: VRPInstance) -> VRPSolution:
        """
        Solves the given VRPInstance, measuring execution time and validating output.

        Args:
            instance (VRPInstance): Problem instance to solve.

        Returns:
            VRPSolution: Standardized solution containing routes, cost, feasibility, and runtime.
        """
        if not isinstance(instance, VRPInstance):
            raise TypeError(f"Expected VRPInstance, got {type(instance).__name__}.")

        # Set random seed prior to execution if specified
        if self.seed is not None:
            np.random.seed(self.seed)

        start_time = time.perf_counter()
        routes = self._solve_instance(instance)
        end_time = time.perf_counter()
        runtime_seconds = end_time - start_time

        # Calculate total distance cost
        total_distance = instance.calculate_total_cost(routes)

        # Compute payload loads and validate feasibility for each route
        vehicle_loads = []
        is_feasible = True

        visited_customers = []
        for route in routes:
            val_res = instance.validate_route(route)
            if not val_res["is_valid"]:
                is_feasible = False

            # Calculate vehicle demand payload
            cust_nodes = [node for node in route if node != instance.depot_idx]
            visited_customers.extend(cust_nodes)
            load = sum(float(instance.demands[node]) for node in cust_nodes)
            vehicle_loads.append(float(load))

        # Check if all customers were visited at least once
        expected_customers = set(instance.customer_indices)
        actual_customers = set(visited_customers)
        if expected_customers != actual_customers:
            is_feasible = False

        return VRPSolution(
            algorithm_name=self.algorithm_name,
            routes=routes,
            total_distance=float(total_distance),
            feasible=is_feasible,
            runtime_seconds=float(runtime_seconds),
            vehicle_loads=vehicle_loads,
            metadata={"seed": self.seed, "params": self.params},
        )


class RandomBaselineSolver(BaseVRPSolver):
    """
    A simple baseline solver that constructs a naive sequential route for testing interface.
    """

    @property
    def algorithm_name(self) -> str:
        return "Random Baseline"

    def _solve_instance(self, instance: VRPInstance) -> List[List[int]]:
        """Constructs a simple sequential route Visiting all customers."""
        route = [instance.depot_idx] + instance.customer_indices + [instance.depot_idx]
        return [route]
