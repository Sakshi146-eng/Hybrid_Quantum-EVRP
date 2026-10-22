import sys
import numpy as np
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.qubo import extract_sub_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver

inst = load_e_cvrp_instance("data/raw/e_cvrp/E-n29-k4-s7.evrp")
sub_inst = extract_sub_instance(inst, customer_count=15)
D = sub_inst.distance_matrix

print("Instance with 15 customers:")
ga = GeneticAlgorithmSolver(population_size=40, generations=30, seed=42)
sol = ga.solve(sub_inst)
route = sol.routes[0]
print(f"GA route: {route}, dist: {sol.total_distance:.4f}")

# Test 2-opt
def two_opt(route, dist_matrix):
    best_route = list(route)
    best_dist = dist_matrix[best_route[0], best_route[1]] + sum(dist_matrix[best_route[i], best_route[i+1]] for i in range(1, len(best_route)-1))
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                # Reverse segment between i and j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = dist_matrix[new_route[0], new_route[1]] + sum(dist_matrix[new_route[k], new_route[k+1]] for k in range(1, len(new_route)-1))
                if new_dist < best_dist - 1e-4:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    return best_route, best_dist

ls_route, ls_dist = two_opt(route, D)
print(f"GA + 2-opt route: {ls_route}, dist: {ls_dist:.4f} (Improvement: {sol.total_distance - ls_dist:.4f})")
