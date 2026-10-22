"""
EVRP Feasibility Checker and Multi-Route Solution Validator.
Verifies vehicle load capacity, battery SoC after every leg, depot start/end,
and unserved customer coverage.
"""
from typing import Dict, List, Tuple, Any
from src.evrp.instance import EVRPInstance


class EVRPFeasibilityChecker:
    """
    Independent validator for Electric Vehicle Routing solution feasibility.
    Evaluates individual vehicle routes as well as complete multi-vehicle solutions.
    """

    def __init__(self, evrp_instance: EVRPInstance):
        """
        Args:
            evrp_instance (EVRPInstance): Configured EVRP problem instance.
        """
        self.instance = evrp_instance

    def check_route_feasibility(self, route: List[int]) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if a single route satisfies all EVRP constraints:
        1. Depot departure and return.
        2. Payload capacity not exceeded.
        3. Battery level never drops below minimum reserve during traversal.

        Args:
            route (List[int]): Ordered list of node indices.

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_feasible, details_dict)
        """
        eval_res = self.instance.evaluate_route_ev(route)
        details = {
            "depot_start_end": eval_res["depot_valid"],
            "capacity": eval_res["capacity_valid"],
            "battery": eval_res["battery_valid"],
            "total_distance": eval_res["total_distance"],
            "total_energy_consumed": eval_res["total_energy_consumed"],
            "total_charging_cost": eval_res["total_charging_cost"],
            "num_charging_stops": eval_res["num_charging_stops"],
            "violations": eval_res["violations"],
            "leg_trace": eval_res["leg_trace"],
        }
        return bool(eval_res["feasible"]), details

    def check_solution_feasibility(
        self, routes: List[List[int]]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks complete multi-vehicle routing solution feasibility:
        1. Every route is individually battery- and capacity-feasible.
        2. Number of routes <= available vehicle fleet.
        3. Every customer node is served exactly once.

        Args:
            routes (List[List[int]]): Set of vehicle routes.

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_feasible, solution_details_dict)
        """
        if not routes:
            return False, {
                "feasible": False,
                "all_routes_feasible": False,
                "customer_coverage": False,
                "vehicle_count_valid": False,
                "unserved_customers": list(self.instance.customer_indices),
                "total_distance": 0.0,
                "total_energy_consumed": 0.0,
                "total_charging_cost": 0.0,
                "violations": ["No routes provided."],
            }

        all_routes_feasible = True
        total_distance = 0.0
        total_energy = 0.0
        total_charging_cost = 0.0
        total_charging_stops = 0
        visited_customers: List[int] = []
        route_evaluations: List[Dict[str, Any]] = []
        violations: List[str] = []

        for r_idx, r in enumerate(routes):
            is_feas, r_details = self.check_route_feasibility(r)
            if not is_feas:
                all_routes_feasible = False
                for v in r_details["violations"]:
                    violations.append(f"Route {r_idx}: {v}")

            total_distance += r_details["total_distance"]
            total_energy += r_details["total_energy_consumed"]
            total_charging_cost += r_details["total_charging_cost"]
            total_charging_stops += r_details["num_charging_stops"]

            custs_in_r = [n for n in r if self.instance.is_customer(n)]
            visited_customers.extend(custs_in_r)
            route_evaluations.append(r_details)

        # Vehicle count check
        vehicle_count_valid = len(routes) <= self.instance.num_vehicles
        if not vehicle_count_valid:
            violations.append(
                f"Used {len(routes)} routes exceeding fleet limit of {self.instance.num_vehicles} vehicles."
            )

        # Customer coverage check
        expected_customers = set(self.instance.customer_indices)
        actual_customers = set(visited_customers)
        unserved = list(expected_customers - actual_customers)
        duplicates = len(visited_customers) - len(actual_customers)

        coverage_valid = (expected_customers == actual_customers) and (duplicates == 0)
        if unserved:
            violations.append(f"Unserved customers ({len(unserved)}): {unserved}")
        if duplicates > 0:
            violations.append(f"Customer nodes visited multiple times ({duplicates} duplicates).")

        overall_feasible = bool(all_routes_feasible and vehicle_count_valid and coverage_valid)

        return overall_feasible, {
            "feasible": overall_feasible,
            "all_routes_feasible": all_routes_feasible,
            "vehicle_count_valid": vehicle_count_valid,
            "customer_coverage": coverage_valid,
            "num_vehicles_used": len(routes),
            "unserved_customers": unserved,
            "total_distance": round(total_distance, 4),
            "total_energy_consumed_kwh": round(total_energy, 4),
            "total_charging_cost_usd": round(total_charging_cost, 4),
            "total_charging_stops": total_charging_stops,
            "route_evaluations": route_evaluations,
            "violations": violations,
        }
