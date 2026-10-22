"""
Investigation script:
1. Loads E-n29-k4-s7.evrp and X-n360-k40-s9.evrp.
2. Extracts 3 and 4-customer sub-instances.
3. Tests QUBO builder, penalty weights, Ising eigenvalues, and QAOA sampling.
4. Identifies why valid sample rate drops to 0% on 4 customers.
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.qubo import extract_sub_instance, VRPQuboBuilder
from src.quantum.qaoa_solver import QAOAVRPSolver
from src.quantum.exact_solver import ExactBruteForceSolver
from src.classical.genetic_algorithm import GeneticAlgorithmSolver

def investigate(instance_file: str, num_cust: int):
    print(f"\n==========================================")
    print(f"Testing {instance_file} with {num_cust} customers")
    print(f"==========================================")
    
    inst = load_e_cvrp_instance(instance_file)
    sub_inst = extract_sub_instance(inst, customer_count=num_cust)
    print(f"Sub-instance customers: {sub_inst.customer_indices}")
    print(f"Distance matrix shape: {sub_inst.distance_matrix.shape}")
    print(f"Max distance: {np.max(sub_inst.distance_matrix):.2f}")
    print(f"Min non-zero distance: {np.min(sub_inst.distance_matrix[sub_inst.distance_matrix > 0]):.2f}")
    
    # Check Exact
    exact = ExactBruteForceSolver().solve(sub_inst)
    print(f"Exact optimal route: {exact.routes[0]}, distance: {exact.total_distance:.4f}")
    
    # Check GA
    ga = GeneticAlgorithmSolver(population_size=50, generations=100, seed=42).solve(sub_inst)
    print(f"GA route: {ga.routes[0]}, distance: {ga.total_distance:.4f}")
    
    # Check default QUBO
    builder = VRPQuboBuilder(sub_inst)
    Q, offset = builder.build_qubo()
    print(f"Default penalty weight: {builder.penalty_weight:.2f}")
    print(f"QUBO offset: {offset:.2f}")
    
    # Evaluate exact route energy vs actual distance
    exact_perm = [c for c in exact.routes[0] if c != 0]
    x_exact = np.zeros(builder.num_vars)
    for p, c in enumerate(exact_perm):
        x_exact[builder.get_var_index(c, p)] = 1.0
    energy_exact = float(x_exact.T @ Q @ x_exact + offset)
    print(f"Exact route QUBO energy: {energy_exact:.4f} (Actual distance: {exact.total_distance:.4f})")
    
    # Check QAOA with current settings
    solver = QAOAVRPSolver(p_depth=1, shots=1024, max_iter=30, seed=42)
    q_sol = solver.solve(sub_inst)
    print(f"QAOA metadata:")
    print(f"  Valid samples: {q_sol.metadata['valid_samples']} / {q_sol.metadata['total_samples']} ({q_sol.metadata['valid_sample_rate']}%)")
    print(f"  Chosen route: {q_sol.routes[0]}")
    print(f"  Chosen distance: {q_sol.total_distance:.4f}")
    print(f"  Best valid bitstring: {q_sol.metadata.get('best_valid_bitstring')}")
    print(f"  Top candidate bitstrings:")
    for cand in q_sol.metadata.get("top_candidates", [])[:5]:
        print(f"    bs: {cand['bitstring']}, count: {cand['count']}, valid: {cand['is_valid']}, dist: {cand['distance']}, energy: {cand['energy']}")

if __name__ == "__main__":
    investigate("data/raw/e_cvrp/E-n29-k4-s7.evrp", 3)
    investigate("data/raw/e_cvrp/E-n29-k4-s7.evrp", 4)
    if Path("data/raw/e_cvrp/X-n360-k40-s9.evrp").exists():
        investigate("data/raw/e_cvrp/X-n360-k40-s9.evrp", 4)
