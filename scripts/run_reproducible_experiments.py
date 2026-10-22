"""
Multi-Seed Reproducible Benchmark Experiment Runner.
Executes Exact, Classical GA, GA + Local Search, QAOA Simulator, and Hybrid GA-QAOA
across multiple random seeds and benchmark instances, recording all metrics.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.comparison import run_quantum_classical_comparison


def run_experiment_suite(
    instance_name: str,
    customer_count: int,
    seeds: List[int],
    shots: int = 1024,
    max_iter: int = 25,
    p_depth: int = 1,
    hybrid_iterations: int = 2,
    subproblem_size: int = 3,
) -> Dict[str, Any]:
    file_path = root_dir / "data" / "raw" / "e_cvrp" / instance_name
    instance = load_e_cvrp_instance(str(file_path))

    results_by_seed = {}

    for s in seeds:
        print(f"--> Running {instance_name} (C={customer_count}) | Seed={s}...")
        t0 = time.time()
        res = run_quantum_classical_comparison(
            instance=instance,
            customer_count=customer_count,
            qaoa_params={
                "p_depth": p_depth,
                "shots": shots,
                "max_iter": max_iter,
                "max_allowed_customers": 4,
            },
            ga_params={
                "population_size": 40,
                "generations": 25,
            },
            hybrid_params={
                "hybrid_iterations": hybrid_iterations,
                "subproblem_size": subproblem_size,
            },
            seed=s,
        )
        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.2f}s")
        results_by_seed[s] = res

    return {
        "instance_name": instance_name,
        "customer_count": customer_count,
        "seeds": seeds,
        "results_by_seed": results_by_seed,
    }


def aggregate_algorithm_metrics(suite: Dict[str, Any], algo_key: str) -> Dict[str, Any]:
    distances = []
    runtimes = []
    feasibles = []
    gaps = []
    valid_rates = []

    for s, res in suite["results_by_seed"].items():
        data = res.get(algo_key)
        if not data or data.get("total_distance") is None:
            continue
        distances.append(data["total_distance"])
        runtimes.append(data["runtime_seconds"])
        feasibles.append(1 if data["feasible"] else 0)
        if data.get("optimality_gap_percent") is not None:
            gaps.append(data["optimality_gap_percent"])
        if "raw_valid_sample_rate_percent" in data:
            valid_rates.append(data["raw_valid_sample_rate_percent"])
        elif "raw_valid_sample_rate" in data:
            valid_rates.append(data["raw_valid_sample_rate"])

    if not distances:
        return {}

    return {
        "best_distance": float(np.min(distances)),
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "std_distance": float(np.std(distances)),
        "best_feasible_rate": float(np.mean(feasibles) * 100.0),
        "mean_runtime": float(np.mean(runtimes)),
        "median_runtime": float(np.median(runtimes)),
        "mean_optimality_gap": float(np.mean(gaps)) if gaps else None,
        "mean_valid_sample_rate": float(np.mean(valid_rates)) if valid_rates else None,
    }


def main():
    print("=" * 80)
    print("REPRODUCIBLE QUANTUM-ASSISTED HYBRID EVRP BENCHMARK EXPERIMENTS")
    print("=" * 80)

    seeds = [42, 101, 777, 999]
    experiments = [
        # Experiment 1: The user's original sub-instance (4 customers on X-n360-k40-s9)
        {"instance": "X-n360-k40-s9.evrp", "c": 4, "subproblem": 3, "iter": 2},
        # Experiment 2: 4 customers on E-n29-k4-s7
        {"instance": "E-n29-k4-s7.evrp", "c": 4, "subproblem": 3, "iter": 2},
        # Experiment 3: Scaling to 6 customers on X-n360-k40-s9 (hybrid subproblems size 3)
        {"instance": "X-n360-k40-s9.evrp", "c": 6, "subproblem": 3, "iter": 2},
    ]

    all_data = []

    for exp in experiments:
        suite = run_experiment_suite(
            instance_name=exp["instance"],
            customer_count=exp["c"],
            seeds=seeds,
            shots=1024,
            max_iter=25,
            p_depth=1,
            hybrid_iterations=exp["iter"],
            subproblem_size=exp["subproblem"],
        )

        print("\n" + "-" * 75)
        print(f"SUMMARY FOR {exp['instance']} (Customers={exp['c']}):")
        print("-" * 75)
        print(f"{'Algorithm':<22} | {'Best Dist':<10} | {'Mean Dist':<10} | {'Std Dist':<9} | {'Mean Gap %':<10} | {'Mean Time':<10} | {'Valid Rate %':<12}")
        print("-" * 75)

        for algo, name in [
            ("exact_baseline", "Exact Brute Force"),
            ("classical_ga", "Classical GA"),
            ("ga_local_search", "GA + Local Search"),
            ("qaoa_simulator", "QAOA Simulator"),
            ("hybrid_ga_qaoa", "Hybrid GA-QAOA"),
        ]:
            agg = aggregate_algorithm_metrics(suite, algo)
            if not agg:
                continue
            b_dist = f"{agg['best_distance']:.4f}"
            m_dist = f"{agg['mean_distance']:.4f}"
            s_dist = f"{agg['std_distance']:.4f}"
            m_gap = f"{agg['mean_optimality_gap']:.2f}%" if agg['mean_optimality_gap'] is not None else "N/A"
            m_time = f"{agg['mean_runtime']:.4f}s"
            v_rate = f"{agg['mean_valid_sample_rate']:.2f}%" if agg['mean_valid_sample_rate'] is not None else "N/A"
            print(f"{name:<22} | {b_dist:<10} | {m_dist:<10} | {s_dist:<9} | {m_gap:<10} | {m_time:<10} | {v_rate:<12}")

        # Seed-by-seed table
        print("\nSeed-by-seed breakdown:")
        print(f"{'Seed':<6} | {'Exact':<11} | {'GA':<11} | {'GA+LS':<11} | {'QAOA':<11} | {'Hybrid':<11} | {'Hybrid vs GA %'}")
        print("-" * 75)
        for s in seeds:
            res_s = suite["results_by_seed"][s]
            ex = f"{res_s['exact_baseline']['total_distance']:.4f}" if res_s.get('exact_baseline') else "N/A"
            ga = f"{res_s['classical_ga']['total_distance']:.4f}"
            gals = f"{res_s['ga_local_search']['total_distance']:.4f}"
            q_val = res_s['qaoa_simulator'].get('total_distance') if res_s.get('qaoa_simulator') else None
            qaoa = f"{q_val:.4f}" if q_val is not None else "N/A (>16Q)"
            hyb = f"{res_s['hybrid_ga_qaoa']['total_distance']:.4f}"
            gain = f"{res_s['hybrid_ga_qaoa']['improvement_over_ga_percent']:.2f}%"
            print(f"{s:<6} | {ex:<11} | {ga:<11} | {gals:<11} | {qaoa:<11} | {hyb:<11} | {gain}")

        all_data.append(suite)

    # Save to json file for artifact and reproducibility
    out_path = root_dir / "benchmark_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nAll reproducible benchmark data saved to: {out_path}")


if __name__ == "__main__":
    main()
