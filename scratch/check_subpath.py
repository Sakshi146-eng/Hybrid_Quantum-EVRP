import sys
import numpy as np
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.qubo import extract_sub_instance

inst = load_e_cvrp_instance("data/raw/e_cvrp/E-n29-k4-s7.evrp")
sub_inst = extract_sub_instance(inst, customer_count=8)
D = sub_inst.distance_matrix

entry = 6
custs = [4, 3, 1, 2]
exit_node = 5

current_cost = D[entry, 4] + D[4, 3] + D[3, 1] + D[1, 2] + D[2, exit_node]
print(f"Current sub-path [6 -> 4 -> 3 -> 1 -> 2 -> 5] cost: {current_cost:.4f}")

# Check all 24 permutations
all_costs = []
for perm in itertools.permutations(custs):
    cost = D[entry, perm[0]] + sum(D[perm[i], perm[i+1]] for i in range(len(perm)-1)) + D[perm[-1], exit_node]
    all_costs.append((cost, perm))

all_costs.sort(key=lambda x: x[0])
best_cost, best_perm = all_costs[0]
print(f"Best sub-path [6 -> {' -> '.join(map(str, best_perm))} -> 5] cost: {best_cost:.4f}")
print(f"Improvement potential: {current_cost - best_cost:.4f} units ({(current_cost - best_cost)/current_cost*100:.2f}%)")
