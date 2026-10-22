import sys
import time
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.e_cvrp_loader import load_evrp_instance
from src.classical.ev_genetic_algorithm import EVGeneticAlgorithmSolver
from src.quantum.adaptive_rl_qaoa import MasterFullHybridEVRPSolver
from src.evrp.cost import calculate_evrp_solution_cost

DATA_DIR = root_dir / "data" / "raw" / "e_cvrp"

print("=" * 70)
print("TESTING SCALABILITY IMPROVEMENTS ON N=4, 8, 12, 20, 50")
print("=" * 70)

for size in [4, 8, 12, 20, 50]:
    file_path = DATA_DIR / ("E-n29-k4-s7.evrp" if size <= 28 else "E-n60-k5-s9.evrp")
    inst = load_evrp_instance(
        file_path,
        customer_count=size,
        battery_capacity=80.0,
        energy_consumption_rate=0.20,
        load_factor=0.20,
        seed=42,
    )

    t0 = time.perf_counter()
    ga_solver = EVGeneticAlgorithmSolver(population_size=30, generations=25, seed=42)
    sol_ga = ga_solver.solve(inst)
    eval_ga = calculate_evrp_solution_cost(inst, sol_ga.routes)
    t_ga = time.perf_counter() - t0

    t1 = time.perf_counter()
    hybrid_solver = MasterFullHybridEVRPSolver(hybrid_iterations=2, seed=42)
    sol_hybrid = hybrid_solver.solve(inst)
    eval_hybrid = calculate_evrp_solution_cost(inst, sol_hybrid.routes)
    t_hybrid = time.perf_counter() - t1

    print(f"\n[N = {size:2d} Customers | Fleet Limit = {inst.num_vehicles}]")
    print(f"  EV-GA:         Cost=${eval_ga['total_cost']:8.2f} | Feas={eval_ga['feasible']} | Veh={eval_ga['num_vehicles_used']} | Time={t_ga:6.2f}s")
    print(f"  Master Hybrid: Cost=${eval_hybrid['total_cost']:8.2f} | Feas={eval_hybrid['feasible']} | Veh={eval_hybrid['num_vehicles_used']} | Time={t_hybrid:6.2f}s")
    if eval_hybrid['violations']:
        print(f"    Violations: {eval_hybrid['violations']}")
