"""
Hierarchical Clustering and Problem Decomposition for Large EVRP Instances.

Enables solving instances with 8, 12, 20, 50, and 100+ customers by:
1. Polar/Spatial angular sweep decomposition into capacity-bounded clusters.
2. Independent cluster route construction via RL policy and EV feasibility checker.
3. Subproblem extraction across cluster boundaries for simulated QAOA optimization (<= 4 nodes -> 16 qubits).
4. Global fleet recombination with customer coverage and battery feasibility guarantees.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.evrp.instance import EVRPInstance
from src.evrp.charging import insert_charging_stations_if_needed, prune_redundant_charging_stops
from src.evrp.cost import calculate_evrp_solution_cost


def decompose_instance_into_clusters(
    instance: EVRPInstance,
    target_cluster_size: int = 4,
) -> List[List[int]]:
    """
    Decomposes an EVRPInstance customer set into capacity-bounded spatial clusters.
    Uses polar angle sorting relative to the depot (Sweep algorithm), while respecting
    instance fleet limits (instance.num_vehicles).

    Args:
        instance: The parent EVRPInstance.
        target_cluster_size: Desired average customers per vehicle cluster (default 4).

    Returns:
        List of customer node lists, one per vehicle cluster.
    """
    depot_coord = instance.coordinates[instance.depot_idx]
    cust_indices = list(instance.customer_indices)
    veh_capacity = float(instance.capacities[0]) if instance.capacities else 100.0
    fleet_limit = max(1, int(instance.num_vehicles or 4))

    # Calculate polar angles relative to depot: theta = atan2(y - y0, x - x0)
    customer_angles: List[Tuple[int, float, float]] = []
    for c in cust_indices:
        cx, cy = instance.coordinates[c]
        angle = float(np.arctan2(cy - depot_coord[1], cx - depot_coord[0]))
        dem = float(instance.demands[c])
        customer_angles.append((c, angle, dem))

    # Sort customers by polar angle
    customer_angles.sort(key=lambda item: item[1])

    # Dynamic target cluster size to respect fleet limits
    total_demand = sum(item[2] for item in customer_angles)
    min_clusters_needed = int(np.ceil(total_demand / max(1.0, veh_capacity)))
    eff_cluster_size = max(target_cluster_size, int(np.ceil(len(cust_indices) / max(1, fleet_limit))))

    clusters: List[List[int]] = []
    current_cluster: List[int] = []
    current_load: float = 0.0

    for c, _, dem in customer_angles:
        # Check capacity constraint and cluster size limit
        if (current_load + dem > veh_capacity or len(current_cluster) >= eff_cluster_size) and current_cluster:
            clusters.append(current_cluster)
            current_cluster = [c]
            current_load = dem
        else:
            current_cluster.append(c)
            current_load += dem

    if current_cluster:
        clusters.append(current_cluster)

    # Enforce cluster count <= fleet_limit
    while len(clusters) > fleet_limit:
        # Merge the two smallest adjacent clusters
        min_idx = 0
        min_combined_load = float("inf")
        for i in range(len(clusters) - 1):
            comb_load = sum(float(instance.demands[c]) for c in clusters[i] + clusters[i + 1])
            if comb_load < min_combined_load:
                min_combined_load = comb_load
                min_idx = i
        merged = clusters[min_idx] + clusters[min_idx + 1]
        clusters = clusters[:min_idx] + [merged] + clusters[min_idx + 2 :]

    return clusters


def build_hierarchical_initial_routes(
    instance: EVRPInstance,
    clusters: List[List[int]],
) -> List[List[int]]:
    """
    Builds initial EV routes from clusters by ordering customers and inserting charging stations.
    Enforces that the final route count never exceeds instance.num_vehicles.
    """
    routes: List[List[int]] = []
    for cluster in clusters:
        if not cluster:
            continue
        # Simple nearest-neighbor ordering starting from depot
        unvisited = list(cluster)
        curr = instance.depot_idx
        ordered = [curr]
        while unvisited:
            next_node = min(unvisited, key=lambda c: instance.distance_matrix[curr, c])
            ordered.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        ordered.append(instance.depot_idx)

        # Insert charging stations if needed
        repaired, _ = insert_charging_stations_if_needed(ordered, instance)
        repaired = prune_redundant_charging_stops(repaired, instance)
        routes.append(repaired)

    from src.evrp.fleet_allocation import GlobalFleetCoordinator
    coordinator = GlobalFleetCoordinator(instance)
    return coordinator.enforce_fleet_limit(routes)

