"""
Test script for Hybrid GA-QAOA solver and GA + Local Search baseline.
"""
import sys
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.qubo import extract_sub_instance, VRPQuboBuilder
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.quantum.exact_solver import ExactBruteForceSolver

def run_test():
    inst = load_e_cvrp_instance("data/raw/e_cvrp/E-n29-k4-s7.evrp")
    
    # Test on 4 customers
    sub4 = extract_sub_instance(inst, customer_count=4)
    exact = ExactBruteForceSolver().solve(sub4)
    ga = GeneticAlgorithmSolver(population_size=30, generations=20, seed=42).solve(sub4)
    print(f"4 Customers:")
    print(f"  Exact: dist={exact.total_distance:.4f}, route={exact.routes[0]}")
    print(f"  GA:    dist={ga.total_distance:.4f}, route={ga.routes[0]}")
    
    # Test on 8 customers
    sub8 = extract_sub_instance(inst, customer_count=8)
    ga8 = GeneticAlgorithmSolver(population_size=30, generations=15, seed=1).solve(sub8)
    print(f"\n8 Customers (Seed 1):")
    print(f"  GA:    dist={ga8.total_distance:.4f}, route={ga8.routes[0]}")

if __name__ == "__main__":
    run_test()
