import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.comparison import run_quantum_classical_comparison

inst = load_e_cvrp_instance("data/raw/e_cvrp/X-n360-k40-s9.evrp")
res = run_quantum_classical_comparison(
    instance=inst,
    customer_count=4,
    qaoa_params={"p_depth": 2, "shots": 1024, "max_iter": 30},
    ga_params={"population_size": 30, "generations": 25},
    seed=42,
)

print("\n--- RESULTS FOR 4 CUSTOMERS ON X-n360-k40-s9.evrp ---")
print("Exact:      ", res["exact_baseline"]["total_distance"], "s:", res["exact_baseline"]["runtime_seconds"])
print("GA:         ", res["classical_ga"]["total_distance"], "s:", res["classical_ga"]["runtime_seconds"], "gap:", res["classical_ga"]["optimality_gap_percent"])
print("GA + LS:    ", res["ga_local_search"]["total_distance"], "s:", res["ga_local_search"]["runtime_seconds"], "gap:", res["ga_local_search"]["optimality_gap_percent"])
print("QAOA:       ", res["qaoa_simulator"]["total_distance"], "s:", res["qaoa_simulator"]["runtime_seconds"], "gap:", res["qaoa_simulator"]["optimality_gap_percent"], "valid%:", res["qaoa_simulator"]["valid_sample_rate_percent"])
print("Hybrid:     ", res["hybrid_ga_qaoa"]["total_distance"], "s:", res["hybrid_ga_qaoa"]["runtime_seconds"], "gap:", res["hybrid_ga_qaoa"]["optimality_gap_percent"])
