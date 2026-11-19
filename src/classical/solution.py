"""
Standardized solution representation for classical VRP solvers.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class VRPSolution:
    """
    Standardized solution container for classical VRP algorithms.

    Attributes:
        algorithm_name (str): Name of the algorithm (e.g., 'Genetic Algorithm', 'Simulated Annealing').
        routes (List[List[int]]): List of vehicle routes. Each route starts and ends at depot.
        total_distance (float): Sum of distances across all vehicle routes.
        feasible (bool): True if all routes satisfy VRP constraints (bounds, depot, capacity).
        runtime_seconds (float): Execution wall-clock time in seconds.
        vehicle_loads (List[float]): Total customer demand payload served by each vehicle route.
        metadata (Dict[str, Any]): Additional stats (e.g. generations, cost curve, parameters).
    """
    algorithm_name: str
    routes: List[List[int]]
    total_distance: float
    feasible: bool
    runtime_seconds: float
    vehicle_loads: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts VRPSolution to a plain JSON-serializable dictionary.
        """
        return {
            "algorithm_name": self.algorithm_name,
            "routes": [list(r) for r in self.routes],
            "total_distance": float(round(self.total_distance, 4)),
            "feasible": bool(self.feasible),
            "runtime_seconds": float(round(self.runtime_seconds, 6)),
            "vehicle_loads": [float(round(l, 4)) for l in self.vehicle_loads],
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """
        Returns a formatted human-readable summary text of the solution.
        """
        lines = [
            f"Algorithm      : {self.algorithm_name}",
            f"Total Distance : {self.total_distance:.2f}",
            f"Feasible       : {'YES' if self.feasible else 'NO'}",
            f"Runtime        : {self.runtime_seconds:.6f} s",
            f"Routes ({len(self.routes)} vehicle(s)):",
        ]
        for i, r in enumerate(self.routes, start=1):
            load = self.vehicle_loads[i - 1] if i - 1 < len(self.vehicle_loads) else 0.0
            route_str = " -> ".join(str(node) for node in r)
            lines.append(f"  Vehicle {i}: {route_str} (Load: {load})")
        return "\n".join(lines)
