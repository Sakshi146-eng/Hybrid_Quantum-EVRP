import sys
import numpy as np
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.qubo import extract_sub_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver

inst = load_e_cvrp_instance("data/raw/e_cvrp/E-n29-k4-s7.evrp")
sub_inst = extract_sub_instance(inst, customer_count=8)
D = sub_inst.distance_matrix

for seed in [1, 7, 42, 99, 123]:
    ga = GeneticAlgorithmSolver(population_size=30, generations=15, seed=seed)
    sol = ga.solve(sub_inst)
    route = sol.routes[0]
    total_d = sol.total_distance
    print(f"\nSeed {seed} GA route: {route}, dist: {total_d:.4f}")
    
    # Check all 3-cust and 4-cust windows
    found_improvement = False
    for m in [3, 4]:
        for k in range(1, len(route) - m):
            u = route[k-1]
            custs = route[k:k+m]
            v = route[k+m]
            curr_c = D[u, custs[0]] + sum(D[custs[i], custs[i+1]] for i in range(len(custs)-1)) + D[custs[-1], v]
            for perm in itertools.permutations(custs):
                cost = D[u, perm[0]] + sum(D[perm[i], perm[i+1]] for i in range(len(perm)-1)) + D[perm[-1], v]
                if cost < curr_c - 1e-4:
                    print(f"  Window {m} at k={k}: {u} -> {custs} -> {v} (cost {curr_c:.2f}) can improve to {perm} (cost {cost:.2f}) -> Gain: {curr_c - cost:.2f}")
                    found_improvement = True
                    break
            if found_improvement:
                break
        if found_improvement:
            break
    if not found_improvement:
        print("  Route has no 3/4-window improvement.")
