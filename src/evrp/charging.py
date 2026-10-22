"""
Charging Station Decision and Insertion Logic for Electric Vehicle Routing.
Implements intelligent station selection, route recharge insertion, and redundant stop pruning.
"""
from typing import List, Tuple, Optional
import numpy as np

from src.evrp.instance import EVRPInstance


def find_best_charging_station(
    from_node: int,
    to_node: int,
    current_battery: float,
    current_load: float,
    instance: EVRPInstance,
) -> Optional[int]:
    """
    Selects the most suitable charging station to insert between from_node and to_node.
    Ensures that the station is reachable from from_node with current battery,
    and that to_node can be reached from the station with full battery.

    Args:
        from_node (int): Source node index.
        to_node (int): Target destination node index.
        current_battery (float): Current battery state of charge (kWh).
        current_load (float): Onboard payload load (units).
        instance (EVRPInstance): Problem instance.

    Returns:
        Optional[int]: Best station node index, or None if no feasible station exists.
    """
    if not instance.charging_stations:
        return None

    D = instance.distance_matrix
    best_station = None
    min_detour = float("inf")

    for s in instance.charging_stations:
        # 1. Reachability from from_node to station s
        e_to_s = instance.calculate_leg_energy(from_node, s, current_load=current_load)
        if current_battery - e_to_s < instance.battery_reserve - 1e-6:
            continue  # Cannot reach station

        # 2. Reachability from station s to to_node with full battery
        e_from_s = instance.calculate_leg_energy(s, to_node, current_load=current_load)
        if instance.battery_capacity - e_from_s < instance.battery_reserve - 1e-6:
            continue  # Cannot reach to_node even with full charge

        # 3. Minimize detour distance
        detour = D[from_node, s] + D[s, to_node] - D[from_node, to_node]
        if detour < min_detour:
            min_detour = detour
            best_station = s

    return best_station


def insert_charging_stations_if_needed(
    route: List[int],
    instance: EVRPInstance,
    max_passes: int = 5,
) -> Tuple[List[int], bool]:
    """
    Simulates route traversal and automatically inserts charging stations whenever
    battery depletion would violate minimum reserve thresholds.
    Uses multi-pass lookahead: if direct insertion between adjacent nodes fails,
    tests inserting earlier in the tour before energy is depleted.

    Args:
        route (List[int]): Route starting and ending at depot_idx.
        instance (EVRPInstance): Configured EVRP problem instance.
        max_passes (int): Maximum iterative insertion passes.

    Returns:
        Tuple[List[int], bool]: (repaired_route, is_battery_feasible)
    """
    if not route or len(route) < 2:
        return list(route), False

    if not instance.charging_stations:
        eval_res = instance.evaluate_route_ev(route)
        return list(route), eval_res["battery_valid"]

    current_route = list(route)

    for pass_num in range(max_passes):
        eval_res = instance.evaluate_route_ev(current_route)
        if eval_res["battery_valid"]:
            # Route is already battery-feasible! Prune any redundant stops
            return prune_redundant_charging_stops(current_route, instance), True

        # Find earliest violation step
        first_violation_idx = None
        for step_info in eval_res["leg_trace"]:
            if step_info["battery_violation"]:
                first_violation_idx = step_info["step"]
                break

        if first_violation_idx is None:
            return current_route, True

        # Candidate insertion locations: try inserting a station before the violation
        # from index first_violation_idx down to 0
        best_candidate_route = None
        min_violations_found = len(eval_res["violations"])
        min_detour = float("inf")

        for insert_pos in range(first_violation_idx + 1, 0, -1):
            u = current_route[insert_pos - 1]
            v = current_route[insert_pos]

            # Get battery and load at u from leg_trace
            if insert_pos - 1 == 0:
                batt_at_u = float(instance.initial_battery)
                load_at_u = float(sum(instance.demands[c] for c in current_route if instance.is_customer(c)))
            else:
                trace_entry = eval_res["leg_trace"][insert_pos - 2]
                batt_at_u = float(trace_entry["battery_after_charge_kwh"])
                load_at_u = float(trace_entry["current_load"])

            # Check all charging stations
            for s in instance.charging_stations:
                if s == u or s == v:
                    continue
                e_to_s = instance.calculate_leg_energy(u, s, current_load=load_at_u)
                if batt_at_u - e_to_s < instance.battery_reserve - 1e-6:
                    continue  # Unreachable

                test_route = current_route[:insert_pos] + [s] + current_route[insert_pos:]
                test_eval = instance.evaluate_route_ev(test_route)
                num_v = len(test_eval["violations"])

                detour = instance.distance_matrix[u, s] + instance.distance_matrix[s, v] - instance.distance_matrix[u, v]

                if num_v < min_violations_found or (num_v == min_violations_found and detour < min_detour):
                    min_violations_found = num_v
                    min_detour = detour
                    best_candidate_route = test_route

                    if num_v == 0:
                        break
            if min_violations_found == 0:
                break

        if best_candidate_route is not None:
            current_route = best_candidate_route
        else:
            # Cannot insert any station that reduces violations
            break

    final_eval = instance.evaluate_route_ev(current_route)
    is_feasible = bool(final_eval["battery_valid"])
    if is_feasible:
        current_route = prune_redundant_charging_stops(current_route, instance)

    return current_route, is_feasible


def prune_redundant_charging_stops(
    route: List[int],
    instance: EVRPInstance,
) -> List[int]:
    """
    Removes redundant charging station stops from a route if the route remains
    fully battery-feasible without them.

    Args:
        route (List[int]): Route with potentially unnecessary charging stations.
        instance (EVRPInstance): Problem instance.

    Returns:
        List[int]: Pruned route.
    """
    if len(route) <= 3:
        return list(route)

    pruned = list(route)
    idx = 1
    while idx < len(pruned) - 1:
        node = pruned[idx]
        if instance.is_charging_station(node):
            # Candidate test: remove this charging station
            candidate = pruned[:idx] + pruned[idx + 1 :]
            eval_res = instance.evaluate_route_ev(candidate)
            if eval_res["battery_valid"]:
                pruned = candidate
                continue
        idx += 1

    return pruned
