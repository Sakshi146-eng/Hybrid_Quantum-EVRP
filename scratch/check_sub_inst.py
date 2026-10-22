import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.qubo import extract_sub_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver

inst = load_e_cvrp_instance("data/raw/e_cvrp/E-n29-k4-s7.evrp")
sub_inst = extract_sub_instance(inst, customer_count=8)
print("Customer indices:", sub_inst.customer_indices)
print("Demands:", sub_inst.demands)
print("Capacity:", sub_inst.vehicle_capacity)

ga = GeneticAlgorithmSolver(population_size=30, generations=20, seed=123)
sol = ga.solve(sub_inst)
print("Routes:", sol.routes)
print("Total dist:", sol.total_distance)
