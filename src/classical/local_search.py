"""
Classical Local Search Operators for Vehicle Routing Problems.
Implements 2-Opt edge inversions, node relocations (insertion moves), and node swaps.
Strictly preserves depot boundaries and route feasibility.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.vrp.instance import VRPInstance


def calculate_route_cost(route: List[int], dist_matrix: np.ndarray) -> float:
    """Calculates total Euclidean travel distance for a single route."""
    if len(route) < 2:
        return 0.0
    cost = 0.0
    for idx in range(len(route) - 1):
        cost += float(dist_matrix[route[idx], route[idx + 1]])
    return cost


def two_opt_single_route(
    route: List[int],
    dist_matrix: np.ndarray,
    depot_idx: int = 0,
    max_passes: int = 50,
) -> Tuple[List[int], float, int]:
    """
    Applies 2-Opt local search to a single route by reversing intermediate segments.
    Depot start and end positions are strictly preserved.

    Args:
        route (List[int]): Route starting and ending at depot_idx.
        dist_matrix (np.ndarray): Instance distance matrix.
        depot_idx (int): Depot node identifier.
        max_passes (int): Maximum iterative improvement passes.

    Returns:
        Tuple[List[int], float, int]: (best_route, best_cost, num_improvements)
    """
    n = len(route)
    if n <= 3:
        return list(route), calculate_route_cost(route, dist_matrix), 0

    best_route = list(route)
    best_cost = calculate_route_cost(best_route, dist_matrix)
    improvements = 0

    for _ in range(max_passes):
        found_improvement = False
        # Intermediate customer indices are in 1 .. n-2
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Candidate route with reversed subsegment [i:j+1]
                new_route = best_route[:i] + best_route[i : j + 1][::-1] + best_route[j + 1 :]
                new_cost = calculate_route_cost(new_route, dist_matrix)

                if new_cost < best_cost - 1e-6:
                    best_route = new_route
                    best_cost = new_cost
                    improvements += 1
                    found_improvement = True
                    break
            if found_improvement:
                break
        if not found_improvement:
            break

    return best_route, best_cost, improvements


def relocate_single_route(
    route: List[int],
    dist_matrix: np.ndarray,
    max_passes: int = 30,
) -> Tuple[List[int], float, int]:
    """
    Applies Relocate (re-insertion) moves: removes customer at index i and inserts at index j.
    """
    n = len(route)
    if n <= 3:
        return list(route), calculate_route_cost(route, dist_matrix), 0

    best_route = list(route)
    best_cost = calculate_route_cost(best_route, dist_matrix)
    improvements = 0

    for _ in range(max_passes):
        found_improvement = False
        for i in range(1, n - 1):
            cust = best_route[i]
            remaining = best_route[:i] + best_route[i + 1 :]
            for j in range(1, len(remaining)):
                candidate = remaining[:j] + [cust] + remaining[j:]
                cost = calculate_route_cost(candidate, dist_matrix)
                if cost < best_cost - 1e-6:
                    best_route = candidate
                    best_cost = cost
                    improvements += 1
                    found_improvement = True
                    break
            if found_improvement:
                break
        if not found_improvement:
            break

    return best_route, best_cost, improvements


def swap_single_route(
    route: List[int],
    dist_matrix: np.ndarray,
    max_passes: int = 30,
) -> Tuple[List[int], float, int]:
    """
    Applies Swap moves: exchanges positions of customer at index i and customer at index j.
    """
    n = len(route)
    if n <= 3:
        return list(route), calculate_route_cost(route, dist_matrix), 0

    best_route = list(route)
    best_cost = calculate_route_cost(best_route, dist_matrix)
    improvements = 0

    for _ in range(max_passes):
        found_improvement = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                candidate = list(best_route)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                cost = calculate_route_cost(candidate, dist_matrix)
                if cost < best_cost - 1e-6:
                    best_route = candidate
                    best_cost = cost
                    improvements += 1
                    found_improvement = True
                    break
            if found_improvement:
                break
        if not found_improvement:
            break

    return best_route, best_cost, improvements


def apply_comprehensive_local_search(
    routes: List[List[int]],
    instance: VRPInstance,
) -> Tuple[List[List[int]], float, int]:
    """
    Applies comprehensive local search (2-Opt, Relocate, Swap) to all routes.
    Ensures that returned routes satisfy instance capacity constraints.

    Args:
        routes (List[List[int]]): Initial set of routes.
        instance (VRPInstance): Problem instance for distance matrix and capacity checks.

    Returns:
        Tuple[List[List[int]], float, int]: (optimized_routes, total_cost, total_improvements)
    """
    dist_matrix = instance.distance_matrix
    depot_idx = instance.depot_idx

    optimized_routes = []
    total_improvements = 0

    for r in routes:
        curr_route = list(r)
        # 1. 2-Opt pass
        r2, _, imp2 = two_opt_single_route(curr_route, dist_matrix, depot_idx=depot_idx)
        # 2. Relocate pass
        r3, _, imp3 = relocate_single_route(r2, dist_matrix)
        # 3. Swap pass
        r4, _, imp4 = swap_single_route(r3, dist_matrix)

        total_improvements += (imp2 + imp3 + imp4)

        # Validate feasibility before accepting
        val = instance.validate_route(r4)
        if val["is_valid"]:
            optimized_routes.append(r4)
        else:
            optimized_routes.append(r)

    total_cost = instance.calculate_total_cost(optimized_routes)
    return optimized_routes, total_cost, total_improvements
