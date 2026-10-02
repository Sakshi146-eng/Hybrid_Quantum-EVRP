"""
EVRP Feasibility Checker (Phase 2).
Verifies route feasibility against vehicle payload capacity, battery energy limits, and charging station visits.
"""
from typing import Dict, List, Tuple
from src.evrp.instance import EVRPInstance


class EVRPFeasibilityChecker:
    """
    Independent validator for EV routing solution feasibility.
    """

    def __init__(self, evrp_instance: EVRPInstance):
        self.instance = evrp_instance

    def check_route_feasibility(self, route: List[int]) -> Tuple[bool, Dict[str, bool]]:
        """
        Checks if a single route satisfies all EVRP constraints:
        1. Payload capacity does not exceed vehicle limit.
        2. Battery level never drops below 0 during traversal.
        3. Route starts and ends at the depot.

        Args:
            route (List[int]): Ordered list of node indices.

        Returns:
            Tuple[bool, Dict[str, bool]]:
                - overall_feasible: True if all checks pass.
                - detail_checks: Dictionary detailing specific constraint outcomes.
        """
        if not route or len(route) < 2:
            return False, {"depot_start_end": False, "capacity": False, "battery": False}

        depot_valid = (route[0] == self.instance.depot_idx) and (route[-1] == self.instance.depot_idx)

        # Capacity check over unique customer nodes visited in the route
        unique_nodes = set(route)
        total_demand = sum(self.instance.demands[node] for node in unique_nodes if node < len(self.instance.demands))
        max_cap = self.instance.capacities[0] if self.instance.capacities else float("inf")
        capacity_valid = total_demand <= max_cap

        # Battery tracking check
        current_battery = self.instance.battery_capacity
        battery_valid = True

        for i in range(len(route) - 1):
            curr_node = route[i]
            next_node = route[i + 1]

            dist = self.instance.distance_matrix[curr_node, next_node]
            energy_needed = dist * self.instance.energy_consumption_rate

            current_battery -= energy_needed

            if current_battery < 0.0 - 1e-6:
                battery_valid = False
                break

            # Recharge if at charging station
            if self.instance.is_charging_station(next_node) or next_node == self.instance.depot_idx:
                current_battery = self.instance.battery_capacity

        overall = bool(depot_valid and capacity_valid and battery_valid)
        details = {
            "depot_start_end": bool(depot_valid),
            "capacity": bool(capacity_valid),
            "battery": bool(battery_valid),
        }
        return overall, details
