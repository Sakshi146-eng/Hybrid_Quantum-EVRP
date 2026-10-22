"""
Unified Comparative Evaluation Workflow for GA and QUBO-QAOA Solvers.
Runs Exact Brute-Force, Classical GA, and Simulated QAOA on the exact same routing instance
and computes standardized comparative metrics (Distance, Runtime, Feasibility, Optimality Gap, Valid-Sample Rate).
"""
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

from src.vrp.instance import VRPInstance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.classical.solution import VRPSolution
from src.quantum.qubo import extract_sub_instance
from src.quantum.qaoa_solver import QAOAVRPSolver
from src.quantum.exact_solver import ExactBruteForceSolver
from src.classical.ga_local_search import GALocalSearchSolver
from src.quantum.hybrid_solver import HybridGAQAOASolver


def run_quantum_classical_comparison(
    instance: VRPInstance,
    customer_indices: Optional[List[int]] = None,
    customer_count: Optional[int] = 3,
    qaoa_params: Optional[Dict[str, Any]] = None,
    ga_params: Optional[Dict[str, Any]] = None,
    hybrid_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = 42,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Executes a fair, equivalent comparison between:
      1. Exact Baseline (Brute-force ground truth)
      2. Classical Genetic Algorithm (GA)
      3. GA + Local Search (Classical Heuristic + 2-Opt)
      4. Simulated QAOA (Qiskit Aer on QUBO)
      5. Hybrid GA-QAOA (Staged Quantum-Classical Optimization)

    All algorithms evaluate the identical sub-instance with identical distance calculations.
    """
    # 1. Extract identical sub-instance
    sub_instance = extract_sub_instance(
        parent_instance=instance,
        customer_indices=customer_indices,
        customer_count=customer_count,
    )

    num_cust = sub_instance.num_customers
    parent_name = (
        instance.raw_data.get("name", "benchmark_instance")
        if hasattr(instance, "raw_data") and isinstance(instance.raw_data, dict)
        else "benchmark_instance"
    )

    # 2. Run Exact Brute-Force Baseline (for <= 8 customers)
    exact_solver = ExactBruteForceSolver()
    exact_solution: VRPSolution = exact_solver.solve(sub_instance)
    exact_optimal_distance = exact_solution.total_distance

    # 3. Run Classical Genetic Algorithm
    g_params = ga_params or {}
    ga_pop = g_params.get("population_size", 50)
    ga_gen = g_params.get("generations", 100)
    ga_cx = g_params.get("crossover_rate", 0.85)
    ga_mut = g_params.get("mutation_rate", 0.25)

    ga_solver = GeneticAlgorithmSolver(
        population_size=ga_pop,
        generations=ga_gen,
        crossover_rate=ga_cx,
        mutation_rate=ga_mut,
        seed=seed,
    )
    ga_solution: VRPSolution = ga_solver.solve(sub_instance)

    # 4. Run Classical GA + Local Search Baseline
    ga_ls_solver = GALocalSearchSolver(
        population_size=ga_pop,
        generations=ga_gen,
        crossover_rate=ga_cx,
        mutation_rate=ga_mut,
        seed=seed,
    )
    ga_ls_solution: VRPSolution = ga_ls_solver.solve(sub_instance)

    # 5. Run Simulated QAOA Solver (standalone direct QUBO is feasible for <= 4 customers)
    q_params = qaoa_params or {}
    q_depth = q_params.get("p_depth", 1)
    q_shots = q_params.get("shots", 1024)
    q_iter = q_params.get("max_iter", 50)

    if num_cust <= 4:
        qaoa_solver = QAOAVRPSolver(
            p_depth=q_depth,
            shots=q_shots,
            max_iter=q_iter,
            seed=seed,
            penalty_multiplier=2.0,
        )
        qaoa_solution: Optional[VRPSolution] = qaoa_solver.solve(sub_instance)
    else:
        qaoa_solution = None

    # 6. Run Hybrid GA-QAOA Solver (uses local subproblems <= 4 customers)
    h_params = hybrid_params or {}
    hybrid_solver = HybridGAQAOASolver(
        ga_population_size=ga_pop,
        ga_generations=ga_gen,
        ga_crossover_rate=ga_cx,
        ga_mutation_rate=ga_mut,
        subproblem_size=min(4, max(2, num_cust)),
        max_hybrid_iterations=h_params.get("max_hybrid_iterations", 3),
        qaoa_p_depth=q_depth,
        qaoa_shots=q_shots,
        qaoa_max_iter=min(40, q_iter),
        use_local_search=True,
        seed=seed,
    )
    hybrid_solution: VRPSolution = hybrid_solver.solve(sub_instance)

    # 7. Compute Comparative Metrics
    def calc_gap(dist: Optional[float]) -> Optional[float]:
        if dist is None or exact_optimal_distance is None or exact_optimal_distance <= 0:
            return None
        return round(((dist - exact_optimal_distance) / exact_optimal_distance) * 100.0, 4)

    def calc_improvement(base_dist: float, new_dist: float) -> float:
        if base_dist <= 1e-6:
            return 0.0
        return round(((base_dist - new_dist) / base_dist) * 100.0, 4)

    ga_gap = calc_gap(ga_solution.total_distance)
    ga_ls_gap = calc_gap(ga_ls_solution.total_distance)
    qaoa_gap = calc_gap(qaoa_solution.total_distance) if qaoa_solution else None
    hybrid_gap = calc_gap(hybrid_solution.total_distance)

    hybrid_improv_over_ga = calc_improvement(ga_solution.total_distance, hybrid_solution.total_distance)
    gals_improv_over_ga = calc_improvement(ga_solution.total_distance, ga_ls_solution.total_distance)

    # Sub-instance coordinates for visualization
    depot_coord = sub_instance.coordinates[0]
    customers_data = []
    for i in range(1, sub_instance.num_nodes):
        c = sub_instance.coordinates[i]
        customers_data.append({
            "id": i,
            "x": float(c[0]),
            "y": float(c[1]),
            "demand": float(sub_instance.demands[i]),
        })

    qaoa_dict = {
        "algorithm": "QAOA Simulator",
        "total_distance": float(round(qaoa_solution.total_distance, 4)) if qaoa_solution else None,
        "feasible": qaoa_solution.feasible if qaoa_solution else False,
        "runtime_seconds": float(round(qaoa_solution.runtime_seconds, 6)) if qaoa_solution else 0.0,
        "optimality_gap_percent": qaoa_gap,
        "routes": qaoa_solution.routes if qaoa_solution else [],
        "p_depth": qaoa_solution.metadata.get("p_depth", q_depth) if qaoa_solution else q_depth,
        "qubits": qaoa_solution.metadata.get("qubits", num_cust**2) if qaoa_solution else num_cust**2,
        "shots": qaoa_solution.metadata.get("shots", q_shots) if qaoa_solution else q_shots,
        "valid_samples": qaoa_solution.metadata.get("valid_samples", 0) if qaoa_solution else 0,
        "total_samples": qaoa_solution.metadata.get("total_samples", q_shots) if qaoa_solution else q_shots,
        "valid_sample_rate_percent": qaoa_solution.metadata.get("valid_sample_rate", 0.0) if qaoa_solution else 0.0,
        "raw_valid_sample_rate_percent": qaoa_solution.metadata.get("raw_valid_sample_rate", 0.0) if qaoa_solution else 0.0,
        "best_raw_sample_distance": qaoa_solution.metadata.get("best_raw_sample_distance") if qaoa_solution else None,
        "best_repaired_distance": qaoa_solution.metadata.get("best_repaired_distance") if qaoa_solution else None,
        "repair_applied": qaoa_solution.metadata.get("repair_applied", False) if qaoa_solution else False,
        "fallback_used": qaoa_solution.metadata.get("fallback_used", False) if qaoa_solution else False,
        "optimizer_evaluations": qaoa_solution.metadata.get("optimizer_evaluations", 0) if qaoa_solution else 0,
        "timing_breakdown": qaoa_solution.metadata.get("timing_breakdown", {}) if qaoa_solution else {},
        "top_candidates": qaoa_solution.metadata.get("top_candidates", []) if qaoa_solution else [],
        "device": "Qiskit Aer Simulator (Classical Emulation)",
        "simulation_note": None if qaoa_solution else f"Whole-instance QAOA simulation requires {num_cust**2} qubits (> 16 qubits), exceeding classical statevector limits. Decomposed into local subproblems in Hybrid GA-QAOA.",
    }

    comparison_record = {
        "benchmark_instance": parent_name,
        "sub_instance_name": sub_instance.raw_data.get("name"),
        "customer_count": num_cust,
        "selected_customer_indices": sub_instance.raw_data.get("selected_customers_parent_ids"),
        "qubits_required": num_cust**2,
        "exact_baseline": {
            "algorithm": exact_solution.algorithm_name,
            "total_distance": float(round(exact_solution.total_distance, 4)),
            "feasible": exact_solution.feasible,
            "runtime_seconds": float(round(exact_solution.runtime_seconds, 6)),
            "routes": exact_solution.routes,
            "evaluated_tours": exact_solution.metadata.get("evaluated_tours", 0),
        },
        "classical_ga": {
            "algorithm": ga_solution.algorithm_name,
            "total_distance": float(round(ga_solution.total_distance, 4)),
            "feasible": ga_solution.feasible,
            "runtime_seconds": float(round(ga_solution.runtime_seconds, 6)),
            "optimality_gap_percent": ga_gap,
            "routes": ga_solution.routes,
            "generations": ga_gen,
            "population_size": ga_pop,
        },
        "ga_local_search": {
            "algorithm": ga_ls_solution.algorithm_name,
            "total_distance": float(round(ga_ls_solution.total_distance, 4)),
            "feasible": ga_ls_solution.feasible,
            "runtime_seconds": float(round(ga_ls_solution.runtime_seconds, 6)),
            "optimality_gap_percent": ga_ls_gap,
            "improvement_over_ga_percent": gals_improv_over_ga,
            "routes": ga_ls_solution.routes,
            "improvements_count": ga_ls_solution.metadata.get("improvements_count", 0),
        },
        "qaoa_simulator": qaoa_dict,
        "hybrid_ga_qaoa": {
            "algorithm": hybrid_solution.algorithm_name,
            "total_distance": float(round(hybrid_solution.total_distance, 4)),
            "feasible": hybrid_solution.feasible,
            "runtime_seconds": float(round(hybrid_solution.runtime_seconds, 6)),
            "optimality_gap_percent": hybrid_gap,
            "improvement_over_ga_percent": hybrid_improv_over_ga,
            "routes": hybrid_solution.routes,
            "qaoa_calls_count": hybrid_solution.metadata.get("qaoa_calls_count", 0),
            "qaoa_improvements_count": hybrid_solution.metadata.get("qaoa_improvements_count", 0),
            "hybrid_improvements_count": hybrid_solution.metadata.get("hybrid_improvements_count", 0),
            "timing_breakdown": hybrid_solution.metadata.get("timing_breakdown", {}),
            "subproblem_logs": hybrid_solution.metadata.get("subproblem_logs", []),
        },
        "visualization_nodes": {
            "depot": {"id": 0, "x": float(depot_coord[0]), "y": float(depot_coord[1])},
            "customers": customers_data,
        },
        "limitations": {
            "model": "Single-vehicle routing without EV charging or battery dynamics",
            "scope": "Phase 1 QUBO-QAOA and Hybrid GA-QAOA foundational comparison",
            "hardware": "Classical simulation via Qiskit Aer (not physical quantum hardware)",
        },
    }

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(comparison_record, f, indent=2)

    return comparison_record
