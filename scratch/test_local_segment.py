"""
Test local subproblem QUBO builder and QAOA optimization on a customer segment.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.vrp.instance import VRPInstance

def test_local_segment_optimization():
    inst = load_e_cvrp_instance("data/raw/e_cvrp/E-n29-k4-s7.evrp")
    # Take 8 customers so GA has a non-trivial problem
    sub_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    sub_dist = inst.distance_matrix[np.ix_(sub_nodes, sub_nodes)]
    sub_demands = inst.demands[sub_nodes]
    sub_coords = inst.coordinates[sub_nodes]
    
    test_inst = VRPInstance(
        distance_matrix=sub_dist,
        coordinates=sub_coords,
        demands=sub_demands,
        vehicle_capacity=500.0,
        num_vehicles=1
    )
    
    # Run GA with small generation count (e.g. 20) to get a realistic heuristic route
    ga = GeneticAlgorithmSolver(population_size=30, generations=20, seed=123)
    ga_sol = ga.solve(test_inst)
    route = ga_sol.routes[0]
    initial_dist = test_inst.calculate_route_distance(route)
    print(f"Initial GA route: {route}, distance: {initial_dist:.4f}")
    
    # Identify segment of 3 or 4 customers with high local cost
    best_gain = 0.0
    best_segment = None
    
    for k in range(1, len(route) - 4):
        # Window of 3 customers: route[k:k+3]
        u = route[k-1] # entry
        segment = route[k:k+3]
        v = route[k+3] # exit
        current_cost = (
            sub_dist[u, segment[0]] +
            sub_dist[segment[0], segment[1]] +
            sub_dist[segment[1], segment[2]] +
            sub_dist[segment[2], v]
        )
        print(f"Segment at k={k}: entry={u}, custs={segment}, exit={v}, cost={current_cost:.2f}")

if __name__ == "__main__":
    test_local_segment_optimization()
