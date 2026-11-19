"""
VRP problem instance representation.
Manages distance matrix, node coordinates, customer demands, vehicle capacities,
and route distance calculation.
"""
from typing import List, Optional, Dict, Any, Union
import numpy as np

from src.vrp.distance import compute_distance_matrix, validate_distance_matrix
from src.vrp.validator import VRPValidator


class VRPInstance:
    """
    Represents a standard classical Vehicle Routing Problem (VRP) instance.
    """

    def __init__(
        self,
        distance_matrix: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
        num_vehicles: int = 1,
        depot_idx: int = 0,
        demands: Optional[np.ndarray] = None,
        vehicle_capacity: Optional[float] = None,
        capacities: Optional[List[float]] = None,
        distance_metric: str = "euclidean",
    ):
        """
        Args:
            distance_matrix (Optional[np.ndarray]): (N, N) distance matrix. If None and coordinates given, computed.
            coordinates (Optional[np.ndarray]): (N, 2) or (N, D) array of node coordinates.
            num_vehicles (int): Number of available vehicles.
            depot_idx (int): Index of the depot node (default 0).
            demands (Optional[np.ndarray]): Customer demands array of size N (depot demand should be 0).
            vehicle_capacity (Optional[float]): Payload capacity for homogeneous vehicle fleet.
            capacities (Optional[List[float]]): List of capacities per vehicle.
            distance_metric (str): Distance metric to use if calculating from coordinates.
        """
        if distance_matrix is None and coordinates is None:
            raise ValueError("Either distance_matrix or coordinates must be provided.")

        if coordinates is not None:
            self.coordinates = np.asarray(coordinates, dtype=float)
            if distance_matrix is None:
                distance_matrix = compute_distance_matrix(self.coordinates, metric=distance_metric)
        else:
            self.coordinates = None

        self.distance_matrix = np.asarray(distance_matrix, dtype=float)

        if self.distance_matrix.ndim != 2 or self.distance_matrix.shape[0] != self.distance_matrix.shape[1]:
            raise ValueError(
                f"Distance matrix must be a 2D square matrix, got shape {self.distance_matrix.shape}."
            )

        self.num_nodes = self.distance_matrix.shape[0]
        self.num_vehicles = num_vehicles
        self.depot_idx = depot_idx
        self.distance_metric = distance_metric

        if not (0 <= depot_idx < self.num_nodes):
            raise ValueError(f"Depot index {depot_idx} out of bounds for {self.num_nodes} nodes.")

        if demands is not None:
            self.demands = np.asarray(demands, dtype=float)
            if len(self.demands) != self.num_nodes:
                raise ValueError(
                    f"Demands array length ({len(self.demands)}) does not match num_nodes ({self.num_nodes})."
                )
        else:
            self.demands = np.zeros(self.num_nodes, dtype=float)

        # Uniform vehicle capacity or per-vehicle capacities list
        if vehicle_capacity is not None:
            self.vehicle_capacity = float(vehicle_capacity)
            self.capacities = [self.vehicle_capacity] * self.num_vehicles
        elif capacities is not None:
            self.capacities = [float(c) for c in capacities]
            self.vehicle_capacity = float(capacities[0]) if capacities else None
        else:
            self.vehicle_capacity = None
            self.capacities = None

        # Validate instance structure
        val_res = VRPValidator.validate_instance_data(
            num_nodes=self.num_nodes,
            depot_idx=self.depot_idx,
            distance_matrix=self.distance_matrix,
            demands=self.demands,
            vehicle_capacity=self.vehicle_capacity
        )
        if not val_res["is_valid"]:
            raise ValueError(f"Invalid VRPInstance data: {'; '.join(val_res['errors'])}")

    @property
    def num_customers(self) -> int:
        """Returns number of customer nodes (total nodes minus depot)."""
        return self.num_nodes - 1

    @property
    def customer_indices(self) -> List[int]:
        """Returns list of customer node indices (excluding depot)."""
        return [i for i in range(self.num_nodes) if i != self.depot_idx]

    def calculate_route_distance(self, route: List[int], validate: bool = True) -> float:
        """
        Calculates total distance of a single ordered route.

        Args:
            route (List[int]): List of node indices in traversal order.
            validate (bool): Whether to perform validation on node indices.

        Returns:
            float: Total distance traveled along the route.

        Raises:
            ValueError: If node index in route is out of bounds [0, num_nodes - 1].
        """
        if not route or len(route) < 2:
            return 0.0

        for node in route:
            if not isinstance(node, (int, np.integer)) or node < 0 or node >= self.num_nodes:
                raise ValueError(
                    f"Invalid node index {node} in route. Valid range is [0, {self.num_nodes - 1}]."
                )

        dist = 0.0
        for i in range(len(route) - 1):
            dist += self.distance_matrix[route[i], route[i + 1]]
        return float(dist)

    def calculate_total_cost(self, routes: List[List[int]], validate: bool = True) -> float:
        """
        Calculates total cost/distance across multiple vehicle routes.

        Args:
            routes (List[List[int]]): List of vehicle routes.
            validate (bool): Whether to perform node index validation.

        Returns:
            float: Sum of distances of all routes.
        """
        return sum(self.calculate_route_distance(r, validate=validate) for r in routes)

    def validate_route(self, route: List[int], check_capacity: bool = True) -> Dict[str, Any]:
        """
        Validates candidate route against VRP rules (depot start/end, node bounds, capacity).

        Args:
            route (List[int]): List of node indices.
            check_capacity (bool): Whether to check vehicle capacity constraint.

        Returns:
            Dict[str, Any]: Validation summary.
        """
        cap = self.vehicle_capacity if check_capacity else None
        return VRPValidator.validate_route(
            route=route,
            num_nodes=self.num_nodes,
            depot_idx=self.depot_idx,
            demands=self.demands,
            vehicle_capacity=cap
        )

    def to_dict(self) -> Dict[str, Any]:
        """Returns instance data as a plain dictionary."""
        return {
            "num_nodes": self.num_nodes,
            "num_customers": self.num_customers,
            "depot_idx": self.depot_idx,
            "num_vehicles": self.num_vehicles,
            "coordinates": self.coordinates.copy() if self.coordinates is not None else None,
            "distance_matrix": self.distance_matrix.copy(),
            "demands": self.demands.copy(),
            "vehicle_capacity": self.vehicle_capacity,
        }
