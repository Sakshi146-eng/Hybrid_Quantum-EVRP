"""
Reproducible EVRP Ablation Study and Scalability Benchmark Runner.
Evaluates all 6 algorithms across multiple random seeds and benchmark instances:
  1. EV-Aware Classical GA
  2. EV-Aware GA + Local Search
  3. Reinforcement Learning (RL)
  4. RL + Classical Local Search
  5. RL + QAOA
  6. Full Hybrid (RL + QAOA + EV-LS)
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.e_cvrp_loader import load_evrp_instance
from src.evrp.benchmark import run_evrp_comparison
from src.rl.agent import EVRLAgent


def run_ablation_experiment_suite(
    instance_name: str,
    customer_count: int,
    seeds: List[int],
    battery_capacity: Optional[float] = None,
) -> Dict[str, Any]:
    file_path = root_dir / "data" / "raw" / "e_cvrp" / instance_name

    suite_results = {}

    for s in seeds:
        print(f"--> Running {instance_name} (C={customer_count}) | Seed={s}...")
        inst = load_evrp_instance(
            file_path=file_path,
            customer_count=customer_count,
            battery_capacity=battery_capacity,
            seed=s,
        )

        rl_agent = EVRLAgent(seed=s)

        res = run_evrp_comparison(
            instance=inst,
            ga_params={"population_size": 30, "generations": 25},
            rl_agent=rl_agent,
            qaoa_params={"p_depth": 1, "shots": 400, "max_iter": 12},
            seed=s,
        )
        suite_results[s] = res

    return {
        "instance_name": instance_name,
        "customer_count": customer_count,
        "seeds": seeds,
        "results_by_seed": suite_results,
    }


def main():
    print("=" * 85)
    print("QUANTUM-ASSISTED EVRP: REPRODUCIBLE ABLATION STUDY & BENCHMARK")
    print("=" * 85)

    seeds = [42, 101, 777]
    configs = [
        {"instance": "E-n29-k4-s7.evrp", "c": 4, "batt": 100.0},
        {"instance": "X-n360-k40-s9.evrp", "c": 6, "batt": 250.0},
    ]

    all_suites = []

    for cfg in configs:
        suite = run_ablation_experiment_suite(
            instance_name=cfg["instance"],
            customer_count=cfg["c"],
            seeds=seeds,
            battery_capacity=cfg["batt"],
        )

        print("\n" + "-" * 85)
        print(f"SUMMARY FOR {cfg['instance']} (Customers={cfg['c']}):")
        print("-" * 85)
        print(f"{'Algorithm':<28} | {'Mean Cost':<10} | {'Mean Dist':<10} | {'Mean kWh':<9} | {'Mean Time':<10} | {'Feasible %'}")
        print("-" * 85)

        algo_keys = ["ga", "ga_ls", "rl", "rl_ls", "rl_qaoa", "full_hybrid"]

        for k in algo_keys:
            costs, dists, energies, times, feas = [], [], [], [], []
            algo_name = ""

            for s in seeds:
                data = suite["results_by_seed"][s]["results"].get(k)
                if not data:
                    continue
                algo_name = data["algorithm_name"]
                costs.append(data["total_cost"])
                dists.append(data["total_distance"])
                energies.append(data["total_energy_kwh"])
                times.append(data["runtime_seconds"])
                feas.append(1 if data["feasible"] else 0)

            if costs:
                m_cost = f"{np.mean(costs):.2f}"
                m_dist = f"{np.mean(dists):.2f}"
                m_eng = f"{np.mean(energies):.2f}"
                m_time = f"{np.mean(times):.4f}s"
                m_feas = f"{np.mean(feas)*100.0:.1f}%"
                print(f"{algo_name:<28} | {m_cost:<10} | {m_dist:<10} | {m_eng:<9} | {m_time:<10} | {m_feas}")

        all_suites.append(suite)

    out_file = root_dir / "evrp_ablation_study_results.json"
    with open(out_file, "w") as f:
        json.dump(all_suites, f, indent=2)

    print(f"\nAblation study benchmark results saved to: {out_file}")


if __name__ == "__main__":
    main()
