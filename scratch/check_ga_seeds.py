import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.qubo import extract_sub_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.quantum.exact_solver import ExactBruteForceSolver

inst = load_e_cvrp_instance("data/raw/e_cvrp/X-n360-k40-s9.evrp")
for num_c in [4, 5, 6]:
    sub = extract_sub_instance(inst, customer_count=num_c)
    exact = ExactBruteForceSolver().solve(sub)
    print(f"\nInstance with {num_c} customers (Exact optimum = {exact.total_distance:.4f}):")
    for s in [1, 7, 42, 99, 123]:
        ga = GeneticAlgorithmSolver(population_size=20, generations=15, seed=s).solve(sub)
        print(f"  Seed {s:3d}: GA dist = {ga.total_distance:.4f}, route = {ga.routes[0]}")
