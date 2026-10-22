"""
Demonstration Script: Quantum vs Classical Optimization Comparison on CEC-2020 E-CVRP Benchmark.
Executes Exact Brute Force (Ground Truth), Classical Genetic Algorithm (GA), and Simulated QAOA (Qiskit Aer)
on an equivalent sub-instance derived from the benchmark dataset.
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.comparison import run_quantum_classical_comparison


def main():
    benchmark_file = root_dir / "data" / "raw" / "e_cvrp" / "E-n29-k4-s7.evrp"
    if len(sys.argv) > 1:
        benchmark_file = Path(sys.argv[1])

    print("=" * 80)
    print(" QUANTUM-ASSISTED EVRP: HYBRID QUANTUM-CLASSICAL BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"Loading CEC-2020 Benchmark Instance: {benchmark_file.name}")

    instance = load_e_cvrp_instance(benchmark_file)

    num_customers = 3
    print(f"Selecting sub-instance: Depot (0) + {num_customers} Customers (1..{num_customers})")
    print(f"QUBO Variables (Qubits Required): {num_customers}^2 = {num_customers**2} qubits")
    print("\nRunning Solvers on Identical Problem Instance:")
    print("  1. Exact Brute Force Baseline (Global Optimum Ground Truth)")
    print("  2. Classical Genetic Algorithm (Permutation GA)")
    print("  3. QAOA Simulator (Qiskit Aer, Depth p=1, 1024 shots, COBYLA optimizer)")
    print("-" * 80)

    results_file = root_dir / "results" / "quantum_classical_comparison.json"
    comp = run_quantum_classical_comparison(
        instance=instance,
        customer_count=num_customers,
        qaoa_params={"p_depth": 1, "shots": 1024, "max_iter": 50},
        ga_params={"population_size": 50, "generations": 100},
        seed=42,
        save_path=results_file,
    )

    exact = comp["exact_baseline"]
    ga = comp["classical_ga"]
    qaoa = comp["qaoa_simulator"]

    print("\n" + "=" * 80)
    print(" COMPARATIVE RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Solver':<22} | {'Distance':<12} | {'Feasible':<10} | {'Opt Gap (%)':<12} | {'Runtime (s)':<12}"
    print(header)
    print("-" * len(header))

    exact_gap_str = "0.00% (Ref)"
    ga_gap_str = f"{ga['optimality_gap_percent']:.2f}%" if ga['optimality_gap_percent'] is not None else "N/A"
    qaoa_gap_str = f"{qaoa['optimality_gap_percent']:.2f}%" if qaoa['optimality_gap_percent'] is not None else "N/A"

    print(f"{exact['algorithm']:<22} | {exact['total_distance']:<12.4f} | {'YES' if exact['feasible'] else 'NO':<10} | {exact_gap_str:<12} | {exact['runtime_seconds']:<12.6f}")
    print(f"{ga['algorithm']:<22} | {ga['total_distance']:<12.4f} | {'YES' if ga['feasible'] else 'NO':<10} | {ga_gap_str:<12} | {ga['runtime_seconds']:<12.6f}")
    print(f"{qaoa['algorithm']:<22} | {qaoa['total_distance']:<12.4f} | {'YES' if qaoa['feasible'] else 'NO':<10} | {qaoa_gap_str:<12} | {qaoa['runtime_seconds']:<12.6f}")
    print("-" * len(header))

    print("\nRoute Sequences:")
    print(f"  Exact Route : {' -> '.join(str(n) for n in exact['routes'][0])}")
    print(f"  GA Route    : {' -> '.join(str(n) for n in ga['routes'][0])}")
    print(f"  QAOA Route  : {' -> '.join(str(n) for n in qaoa['routes'][0])}")

    print("\nQAOA Execution Details (Simulated via Qiskit Aer):")
    print(f"  Qubits                 : {qaoa['qubits']}")
    print(f"  Circuit Depth (p)      : {qaoa['p_depth']}")
    print(f"  Measurement Shots      : {qaoa['shots']}")
    print(f"  Valid Candidate Samples: {qaoa['valid_samples']} / {qaoa['total_samples']} ({qaoa['valid_sample_rate_percent']}%)")
    print(f"  COBYLA Evaluations     : {qaoa['optimizer_evaluations']}")
    if "timing_breakdown" in qaoa:
        tb = qaoa["timing_breakdown"]
        print(f"  Timing Breakdown:")
        print(f"    - Circuit Prep       : {tb.get('circuit_prep_seconds', 0.0):.6f} s")
        print(f"    - COBYLA Optimization: {tb.get('optimization_seconds', 0.0):.6f} s")
        print(f"    - Aer Sampling       : {tb.get('sampling_seconds', 0.0):.6f} s")
        print(f"    - Total Wall-Clock   : {tb.get('total_seconds', 0.0):.6f} s")

    print("\nResearch Disclaimers:")
    print("  * Device: Classical CPU simulation via Qiskit Aer (No physical quantum hardware was used).")
    print("  * Scope: Phase 1 single-vehicle routing without EV charging or battery dynamics.")
    print(f"  * Results recorded to: {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
