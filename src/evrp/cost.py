"""
Standardized Objective Cost Function and Evaluation for EVRP.
Computes combined operational routing cost:
  Total_Cost = w_dist * Distance + w_charging * Charging_Cost + w_vehicle * Num_Vehicles + Infeasibility_Penalty
Ensures 100% fair evaluation across Classical GA, Local Search, RL, QAOA, and Hybrid methods.
"""
from typing import List, Dict, Any, Optional
from src.evrp.instance import EVRPInstance
from src.evrp.feasibility import EVRPFeasibilityChecker


def calculate_evrp_solution_cost(
    instance: EVRPInstance,
    routes: List[List[int]],
    *,
    w_dist: float = 1.0,
    w_charging: float = 1.0,
    w_vehicle: float = 50.0,
    penalty_infeasible: float = 10000.0,
) -> Dict[str, Any]:
    """
    Evaluates a complete multi-vehicle routing solution on an EVRPInstance.
    Computes distance, energy consumption, charging fees, vehicle usage,
    and penalizes constraint violations.

    Args:
        instance (EVRPInstance): Configured problem instance.
        routes (List[List[int]]): List of vehicle routes.
        w_dist (float): Weight for total travel distance.
        w_charging (float): Weight for electricity charging cost ($).
        w_vehicle (float): Fixed dispatch cost per vehicle used ($).
        penalty_infeasible (float): Penalty incurred per constraint violation.

    Returns:
        Dict[str, Any]: Detailed cost breakdown and feasibility summary.
    """
    checker = EVRPFeasibilityChecker(instance)
    is_feasible, details = checker.check_solution_feasibility(routes)

    total_dist = details["total_distance"]
    total_energy = details["total_energy_consumed_kwh"]
    total_charging = details["total_charging_cost_usd"]
    num_vehicles = details["num_vehicles_used"]
    violations = details["violations"]

    dist_cost = w_dist * total_dist
    ch_cost = w_charging * total_charging
    veh_cost = w_vehicle * num_vehicles
    pen_cost = penalty_infeasible * len(violations) if not is_feasible else 0.0

    total_cost = dist_cost + ch_cost + veh_cost + pen_cost

    return {
        "total_cost": round(float(total_cost), 4),
        "distance_cost": round(float(dist_cost), 4),
        "charging_cost": round(float(ch_cost), 4),
        "vehicle_cost": round(float(veh_cost), 4),
        "penalty_cost": round(float(pen_cost), 4),
        "feasible": is_feasible,
        "total_distance": round(float(total_dist), 4),
        "total_energy_consumed_kwh": round(float(total_energy), 4),
        "total_charging_cost_usd": round(float(total_charging), 4),
        "total_charging_stops": details["total_charging_stops"],
        "num_vehicles_used": num_vehicles,
        "unserved_customers": details["unserved_customers"],
        "violations": violations,
        "route_evaluations": details["route_evaluations"],
    }
