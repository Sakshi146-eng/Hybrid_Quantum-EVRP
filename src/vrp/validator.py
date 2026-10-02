"""
Validation utilities for VRP instances and candidate routes.
Enforces classical VRP constraints without any QUBO/quantum penalty formulations.
"""
from typing import Dict, List, Any, Optional
import numpy as np


class VRPValidator:
    """
    Validates VRP instance data and candidate vehicle routes against classical constraints.
    """

    @staticmethod
    def validate_instance_data(
        num_nodes: int,
        depot_idx: int,
        distance_matrix: np.ndarray,
        demands: Optional[np.ndarray] = None,
        vehicle_capacity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validates basic structure and consistency of a VRP instance.

        Returns:
            Dict[str, Any]: Validation status and error messages.
        """
        errors = []

        if num_nodes <= 1:
            errors.append(f"Number of nodes must be > 1, got {num_nodes}.")

        if not (0 <= depot_idx < num_nodes):
            errors.append(f"Depot index {depot_idx} is out of bounds for {num_nodes} nodes.")

        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != num_nodes or distance_matrix.shape[1] != num_nodes:
            errors.append(f"Distance matrix shape must be ({num_nodes}, {num_nodes}), got {distance_matrix.shape}.")

        if demands is not None:
            if len(demands) != num_nodes:
                errors.append(f"Demands length ({len(demands)}) does not match num_nodes ({num_nodes}).")
            if demands[depot_idx] != 0:
                errors.append(f"Depot demand must be 0, got {demands[depot_idx]}.")
            if np.any(demands < 0):
                errors.append("Demands must be non-negative.")

        if vehicle_capacity is not None and vehicle_capacity <= 0:
            errors.append(f"Vehicle capacity must be positive, got {vehicle_capacity}.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }

    @staticmethod
    def validate_route(
        route: List[int],
        num_nodes: int,
        depot_idx: int = 0,
        demands: Optional[np.ndarray] = None,
        vehicle_capacity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validates a single candidate route.

        Args:
            route (List[int]): List of node indices in traversal order.
            num_nodes (int): Total number of nodes in the VRP instance.
            depot_idx (int): Depot node index.
            demands (Optional[np.ndarray]): Array of customer demands.
            vehicle_capacity (Optional[float]): Maximum payload capacity per vehicle.

        Returns:
            Dict[str, Any]: Detailed validation result dictionary.
        """
        errors = []
        details: Dict[str, Any] = {
            "is_valid": True,
            "errors": errors,
            "node_count": len(route),
            "depot_start_end": False,
            "valid_indices": True,
            "total_demand": 0.0,
            "capacity_valid": True,
        }

        if not route:
            errors.append("Route is empty.")
            details["is_valid"] = False
            return details

        # 1. Check valid node indices
        for idx in route:
            if not isinstance(idx, (int, np.integer)):
                errors.append(f"Route node index must be an integer, got {type(idx).__name__}.")
                details["valid_indices"] = False
                break
            if idx < 0 or idx >= num_nodes:
                errors.append(f"Node index {idx} out of valid range [0, {num_nodes - 1}].")
                details["valid_indices"] = False

        if not details["valid_indices"]:
            details["is_valid"] = False
            return details

        # 2. Check depot start and end
        if len(route) < 2:
            errors.append("Route must contain at least 2 nodes (e.g. [depot, customer] or [depot, depot]).")
            details["depot_start_end"] = False
        elif route[0] != depot_idx or route[-1] != depot_idx:
            errors.append(f"Route must start and end at depot (index {depot_idx}), got start={route[0]}, end={route[-1]}.")
            details["depot_start_end"] = False
        else:
            details["depot_start_end"] = True

        # 3. Check payload capacity constraint
        if demands is not None:
            # Sum customer demands (excluding depot)
            customer_nodes = [node for node in route if node != depot_idx]
            total_demand = float(sum(demands[node] for node in customer_nodes))
            details["total_demand"] = total_demand

            if vehicle_capacity is not None:
                if total_demand > vehicle_capacity:
                    errors.append(
                        f"Route total demand ({total_demand}) exceeds vehicle capacity ({vehicle_capacity})."
                    )
                    details["capacity_valid"] = False

        details["is_valid"] = len(errors) == 0
        return details
