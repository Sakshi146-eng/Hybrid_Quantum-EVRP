import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.vrp.instance import VRPInstance

inst = load_e_cvrp_instance("data/raw/e_cvrp/E-n29-k4-s7.evrp")
sub_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sub_dist = inst.distance_matrix[np.ix_(sub_nodes, sub_nodes)]
sub_demands = inst.demands[sub_nodes]
sub_coords = inst.coordinates[sub_nodes]

test_inst = VRPInstance(
    distance_matrix=sub_dist,
    coordinates=sub_coords,
    demands=sub_demands,
    vehicle_capacity=1000.0,
    num_vehicles=1
)

ga = GeneticAlgorithmSolver(population_size=30, generations=20, seed=123)
ga_sol = ga.solve(test_inst)
print("Routes:", ga_sol.routes)
print("Total distance:", ga_sol.total_distance)
