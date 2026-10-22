"""
EV-Aware Local Search Operators and Solver.
Implements 2-Opt edge inversions, customer relocations, and customer swaps
while dynamically updating charging station insertions and verifying leg-by-leg battery feasibility.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.evrp.instance import EVRPInstance
from src.evrp.charging import insert_charging_stations_if_needed, prune_redundant_charging_stops
from src.evrp.cost import calculate_evrp_solution_cost
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution


def apply_ev_local_search_to_route(
    route: List[int],
    instance: EVRPInstance,
    max_passes: int = 30,
) -> Tuple[List[int], Dict[str, Any], int]:
    """
    Applies 2-Opt and Relocate moves to customer nodes in a route.
    After each structural permutation, charging stations are updated and
    the move is accepted if and only if it strictly improves total EV cost.

    Args:
        route (List[int]): Route starting and ending at depot_idx.
        instance (EVRPInstance): Problem instance.
        max_passes (int): Iteration limit.

    Returns:
        Tuple[List[int], Dict[str, Any], int]: (best_route, best_eval, num_improvements)
    """
    # Clean customer sequence (strip existing charging stations for neighborhood exploration)
    cust_seq = [n for n in route if instance.is_customer(n)]
    if len(cust_seq) < 2:
        repaired, _ = insert_charging_stations_if_needed(route, instance)
        eval_res = instance.evaluate_route_ev(repaired)
        return repaired, eval_res, 0

    best_cust_seq = list(cust_seq)
    curr_raw = [instance.depot_idx] + best_cust_seq + [instance.depot_idx]
    best_route, _ = insert_charging_stations_if_needed(curr_raw, instance)
    best_eval = instance.evaluate_route_ev(best_route)
    best_cost = best_eval["total_distance"] + 1.0 * best_eval["total_charging_cost"] + (0.0 if best_eval["feasible"] else 50000.0)

    improvements = 0

    # For exactly 2 customers, test reversing the pair
    if len(best_cust_seq) == 2:
        rev_seq = [best_cust_seq[1], best_cust_seq[0]]
        cand_raw = [instance.depot_idx] + rev_seq + [instance.depot_idx]
        cand_route, _ = insert_charging_stations_if_needed(cand_raw, instance)
        cand_eval = instance.evaluate_route_ev(cand_route)
        cand_cost = cand_eval["total_distance"] + 1.0 * cand_eval["total_charging_cost"] + (0.0 if cand_eval["feasible"] else 50000.0)
        if cand_cost < best_cost - 1e-4:
            return cand_route, cand_eval, 1
        return best_route, best_eval, 0

    for _ in range(max_passes):
        found_improvement = False
        n_cust = len(best_cust_seq)

        # 1. 2-Opt edge inversion on customer sequence
        for i in range(n_cust - 1):
            for j in range(i + 1, n_cust):
                cand_seq = best_cust_seq[:i] + best_cust_seq[i : j + 1][::-1] + best_cust_seq[j + 1 :]
                cand_raw = [instance.depot_idx] + cand_seq + [instance.depot_idx]
                cand_route, _ = insert_charging_stations_if_needed(cand_raw, instance)
                cand_eval = instance.evaluate_route_ev(cand_route)

                cand_cost = cand_eval["total_distance"] + 1.0 * cand_eval["total_charging_cost"] + (0.0 if cand_eval["feasible"] else 50000.0)

                if cand_cost < best_cost - 1e-4:
                    best_cust_seq = cand_seq
                    best_route = cand_route
                    best_eval = cand_eval
                    best_cost = cand_cost
                    improvements += 1
                    found_improvement = True
                    break
            if found_improvement:
                break

        # 2. Relocate (insertion move)
        if not found_improvement:
            for i in range(n_cust):
                c = best_cust_seq[i]
                rem = best_cust_seq[:i] + best_cust_seq[i + 1 :]
                for j in range(len(rem) + 1):
                    cand_seq = rem[:j] + [c] + rem[j:]
                    cand_raw = [instance.depot_idx] + cand_seq + [instance.depot_idx]
                    cand_route, _ = insert_charging_stations_if_needed(cand_raw, instance)
                    cand_eval = instance.evaluate_route_ev(cand_route)

                    cand_cost = cand_eval["total_distance"] + 1.0 * cand_eval["total_charging_cost"] + (0.0 if cand_eval["feasible"] else 50000.0)

                    if cand_cost < best_cost - 1e-4:
                        best_cust_seq = cand_seq
                        best_route = cand_route
                        best_eval = cand_eval
                        best_cost = cand_cost
                        improvements += 1
                        found_improvement = True
                        break
                if found_improvement:
                    break

        if not found_improvement:
            break

    return best_route, best_eval, improvements


class EVLocalSearchSolver(BaseVRPSolver):
    """
    EV-Aware Local Search Solver.
    Polishes routes using intra-route 2-Opt and Relocate moves, as well as
    inter-route customer relocations, swaps, and route consolidations.
    """

    def __init__(self, seed: Optional[int] = 42, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "EV-Aware Local Search"

    def optimize_routes(
        self, initial_routes: List[List[int]], instance: EVRPInstance
    ) -> Tuple[List[List[int]], int]:
        """
        Polishes a multi-vehicle solution using intra-route and inter-route EV local search.
        """
        veh_capacity = instance.capacities[0] if instance.capacities else 100000.0

        # Phase 1: Intra-route local search on each route
        current_routes = []
        total_imp = 0
        for r in initial_routes:
            opt_r, _, imp = apply_ev_local_search_to_route(r, instance)
            current_routes.append(opt_r)
            total_imp += imp

        best_cost_eval = calculate_evrp_solution_cost(instance, current_routes)
        best_cost = best_cost_eval["total_cost"]

        # Phase 2: Inter-route neighborhood exploration
        max_inter_passes = 6 if instance.num_customers > 30 else 12
        for _ in range(max_inter_passes):
            improved = False

            # Operator A: Route Merging / Consolidation
            if len(current_routes) > 1:
                for i in range(len(current_routes)):
                    for j in range(i + 1, len(current_routes)):
                        r_i = current_routes[i]
                        r_j = current_routes[j]
                        c_i = [n for n in r_i if instance.is_customer(n)]
                        c_j = [n for n in r_j if instance.is_customer(n)]
                        total_dem = sum(float(instance.demands[c]) for c in c_i + c_j)

                        if total_dem <= veh_capacity:
                            # Try merging in forward or reverse orientations
                            for merge_order in [c_i + c_j, c_j + c_i, c_i + c_j[::-1]]:
                                raw_merged = [instance.depot_idx] + merge_order + [instance.depot_idx]
                                rep_merged, _ = insert_charging_stations_if_needed(raw_merged, instance)
                                opt_merged, _, m_imp = apply_ev_local_search_to_route(rep_merged, instance)

                                cand_routes = [
                                    current_routes[k] for k in range(len(current_routes)) if k != i and k != j
                                ] + [opt_merged]

                                cand_eval = calculate_evrp_solution_cost(instance, cand_routes)
                                if cand_eval["feasible"] and cand_eval["total_cost"] < best_cost - 1e-4:
                                    current_routes = cand_routes
                                    best_cost = cand_eval["total_cost"]
                                    best_cost_eval = cand_eval
                                    total_imp += 1 + m_imp
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break

            if improved:
                continue

            # Operator B: Inter-route Customer Relocation
            if len(current_routes) > 1:
                for i in range(len(current_routes)):
                    c_i = [n for n in current_routes[i] if instance.is_customer(n)]
                    for c_idx, cust in enumerate(c_i):
                        for j in range(len(current_routes)):
                            if i == j:
                                continue
                            c_j = [n for n in current_routes[j] if instance.is_customer(n)]
                            dem_cust = float(instance.demands[cust])
                            load_j = sum(float(instance.demands[c]) for c in c_j)

                            if load_j + dem_cust > veh_capacity:
                                continue

                            rem_i = c_i[:c_idx] + c_i[c_idx + 1 :]
                            for pos in range(len(c_j) + 1):
                                ins_j = c_j[:pos] + [cust] + c_j[pos:]

                                cand_r_i = [instance.depot_idx] + rem_i + [instance.depot_idx] if rem_i else None
                                cand_r_j = [instance.depot_idx] + ins_j + [instance.depot_idx]

                                rep_i, _ = insert_charging_stations_if_needed(cand_r_i, instance) if cand_r_i else (None, True)
                                rep_j, _ = insert_charging_stations_if_needed(cand_r_j, instance)

                                cand_routes = []
                                for k in range(len(current_routes)):
                                    if k == i:
                                        if rep_i:
                                            cand_routes.append(rep_i)
                                    elif k == j:
                                        cand_routes.append(rep_j)
                                    else:
                                        cand_routes.append(current_routes[k])

                                cand_eval = calculate_evrp_solution_cost(instance, cand_routes)
                                if cand_eval["feasible"] and cand_eval["total_cost"] < best_cost - 1e-4:
                                    current_routes = cand_routes
                                    best_cost = cand_eval["total_cost"]
                                    best_cost_eval = cand_eval
                                    total_imp += 1
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            if improved:
                continue

            # Operator C: Inter-route Customer Swap
            if len(current_routes) > 1:
                for i in range(len(current_routes) - 1):
                    c_i = [n for n in current_routes[i] if instance.is_customer(n)]
                    load_i_base = sum(float(instance.demands[c]) for c in c_i)
                    for j in range(i + 1, len(current_routes)):
                        c_j = [n for n in current_routes[j] if instance.is_customer(n)]
                        load_j_base = sum(float(instance.demands[c]) for c in c_j)

                        for idx_i, cust_i in enumerate(c_i):
                            dem_i = float(instance.demands[cust_i])
                            for idx_j, cust_j in enumerate(c_j):
                                dem_j = float(instance.demands[cust_j])

                                new_load_i = load_i_base - dem_i + dem_j
                                new_load_j = load_j_base - dem_j + dem_i
                                if new_load_i > veh_capacity or new_load_j > veh_capacity:
                                    continue

                                # Create swapped sequences
                                swap_i = list(c_i)
                                swap_i[idx_i] = cust_j
                                swap_j = list(c_j)
                                swap_j[idx_j] = cust_i

                                cand_r_i = [instance.depot_idx] + swap_i + [instance.depot_idx]
                                cand_r_j = [instance.depot_idx] + swap_j + [instance.depot_idx]

                                rep_i, feas_i = insert_charging_stations_if_needed(cand_r_i, instance)
                                rep_j, feas_j = insert_charging_stations_if_needed(cand_r_j, instance)

                                cand_routes = [list(r) for r in current_routes]
                                cand_routes[i] = rep_i
                                cand_routes[j] = rep_j

                                cand_eval = calculate_evrp_solution_cost(instance, cand_routes)
                                if cand_eval["feasible"] and cand_eval["total_cost"] < best_cost - 1e-4:
                                    current_routes = cand_routes
                                    best_cost = cand_eval["total_cost"]
                                    best_cost_eval = cand_eval
                                    total_imp += 1
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            if improved:
                continue

            # Operator D: Route Merging (Consolidation to save vehicle dispatch cost)
            if len(current_routes) > 1:
                for i in range(len(current_routes) - 1):
                    c_i = [n for n in current_routes[i] if instance.is_customer(n)]
                    load_i = sum(float(instance.demands[c]) for c in c_i)
                    for j in range(i + 1, len(current_routes)):
                        c_j = [n for n in current_routes[j] if instance.is_customer(n)]
                        load_j = sum(float(instance.demands[c]) for c in c_j)

                        if load_i + load_j <= veh_capacity:
                            merged_custs = c_i + c_j
                            cand_merged = [instance.depot_idx] + merged_custs + [instance.depot_idx]
                            rep_merged, is_ok = insert_charging_stations_if_needed(cand_merged, instance)
                            rep_merged = prune_redundant_charging_stops(rep_merged, instance)

                            cand_routes = [current_routes[k] for k in range(len(current_routes)) if k != i and k != j]
                            cand_routes.append(rep_merged)

                            cand_eval = calculate_evrp_solution_cost(instance, cand_routes)
                            if cand_eval["feasible"] and cand_eval["total_cost"] < best_cost - 1e-4:
                                current_routes = cand_routes
                                best_cost = cand_eval["total_cost"]
                                best_cost_eval = cand_eval
                                total_imp += 1
                                improved = True
                                break
                    if improved:
                        break

            if not improved:
                break

        # Final station pruning and route cleaning
        clean_routes = []
        for r in current_routes:
            custs = [n for n in r if instance.is_customer(n)]
            if custs:
                p = prune_redundant_charging_stops(r, instance)
                clean_routes.append(p)

        return clean_routes, total_imp

    def _solve_instance(self, instance: EVRPInstance) -> List[List[int]]:
        from src.evrp.fleet_allocation import build_fleet_coordinated_routes
        initial_routes = build_fleet_coordinated_routes(instance)
        polished_routes, imp = self.optimize_routes(initial_routes, instance)
        self.last_metadata = {"improvements_count": imp}
        return polished_routes


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
