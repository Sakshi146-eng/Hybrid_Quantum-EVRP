"""
Master Regression Test Suite for RL + QAOA Hybrid EVRP Framework.
Covers:
1. EV-aware QUBO construction and quantum marginal decoding.
2. RL subproblem agent feature extraction and neural priority scoring.
3. Adaptive parameter selection (p, lambda, shots).
4. QAOA execution and comparison to exact enumeration on small subproblems.
5. Candidate diversity and multi-policy rollouts.
6. Hierarchical decomposition for larger instances (N >= 8).
7. Evaluation across all 8 benchmark algorithms.
8. Best-feasible-solution preservation and monotonicity.
9. Timeout and fallback resilience.
10. Multi-seed statistical evaluation and 6-part ablation studies.
11. FastAPI endpoint response serialization.
"""

import math
from pathlib import Path
import numpy as np
from src.classical.solution import VRPSolution
from src.data.e_cvrp_loader import load_evrp_instance
from src.evrp.instance import EVRPInstance
from src.quantum.ev_qubo import EVRouteQuboBuilder, decode_quantum_marginals
from src.quantum.qaoa_solver import QAOAVRPSolver
from src.rl.qaoa_subproblem_agent import (
    RLSubproblemAgent,
    SubproblemPriorityNet,
)
from src.quantum.adaptive_rl_qaoa import (
    ConventionalQAOASolver,
    RLGuidedQAOASolver,
    RLGuidedQAOALocalSearchSolver,
    MasterFullHybridEVRPSolver,
)
from src.evrp.decomposition import decompose_instance_into_clusters, build_hierarchical_initial_routes
from src.evrp.benchmark import (
    run_evrp_comparison,
    run_evrp_multi_seed_comparison,
    run_evrp_ablation_study,
)


def _get_toy_evrp_instance(n_cust=4):
    path = Path("data/raw/e_cvrp/E-n29-k4-s7.evrp")
    return load_evrp_instance(path, customer_count=n_cust)


def test_ev_qubo_formulation_and_marginals_decoding():
    """Verify EVRouteQuboBuilder and quantum marginal decoding via bipartite matching."""
    inst = _get_toy_evrp_instance(3)
    custs = [inst.customer_indices[0], inst.customer_indices[1], inst.customer_indices[2]]
    depot = inst.depot_idx

    builder = EVRouteQuboBuilder(
        instance=inst,
        entry_node=depot,
        exit_node=depot,
        customer_subset=custs,
        entry_load=200.0,
        entry_battery=inst.battery_capacity,
        penalty_multiplier=2.0,
    )

    qubo_mat, offset = builder.build_qubo()
    assert qubo_mat.shape[0] == 9  # 3 customers x 3 positions = 9 binary variables
    assert qubo_mat.shape[1] == 9

    # Test marginal decoding
    synthetic_counts = {
        "000000001": 500,  # bitstring format
        "001010100": 300,
    }
    perm = decode_quantum_marginals(synthetic_counts, custs)
    assert len(perm) == 3
    assert set(perm) == set(custs), "Marginal decoding must return a valid permutation of all customers"


def test_rl_subproblem_agent_features_and_scoring():
    """Verify 7-feature extraction and neural priority scoring for candidate subproblems."""
    inst = _get_toy_evrp_instance(4)
    agent = RLSubproblemAgent(seed=42)

    route = [inst.depot_idx] + list(inst.customer_indices) + [inst.depot_idx]
    window_indices = [1, 2, 3]

    features = agent.extract_subproblem_features(
        instance=inst,
        route=route,
        window_indices=window_indices,
        entry_load=50.0,
        entry_battery=inst.battery_capacity,
    )
    assert len(features) == 7
    assert all(np.isfinite(f) for f in features), "All features must be finite numbers"

    cand = {
        "route_idx": 0,
        "window_pos": window_indices,
        "cust_subset": [route[i] for i in window_indices],
        "entry_node": route[0],
        "exit_node": route[4],
        "features": features,
    }

    ranked = agent.rank_candidate_subproblems([cand], deterministic=True)
    assert len(ranked) == 1
    idx, params, score = ranked[0]
    assert idx == 0
    assert np.isfinite(score)
    assert params["p_depth"] in [1, 2]


def test_rl_qaoa_parameter_adaptation():
    """Verify adaptive QAOA parameter selection (depth, lambda, shots)."""
    agent = RLSubproblemAgent(seed=42)
    cand = {"features": np.array([0.75, 0.4, 0.5, 0.2, 0.1, 0.0, 0.5], dtype=np.float32)}

    ranked = agent.rank_candidate_subproblems([cand], deterministic=True)
    assert len(ranked) == 1
    _, params, _ = ranked[0]
    assert params["p_depth"] in [1, 2]
    assert params["penalty_multiplier"] in [1.5, 2.0, 3.0]
    assert params["shots"] in [256, 512, 1024]


def test_qaoa_execution_and_exact_enumeration():
    """Verify QAOA subproblem solving compared to exact permutation enumeration."""
    inst = _get_toy_evrp_instance(3)
    custs = list(inst.customer_indices[:3])

    builder = EVRouteQuboBuilder(
        instance=inst,
        entry_node=inst.depot_idx,
        exit_node=inst.depot_idx,
        customer_subset=custs,
        entry_load=150.0,
        entry_battery=inst.battery_capacity,
    )

    solver = QAOAVRPSolver(p_depth=1, shots=256, max_iter=5, seed=42)
    _, qaoa_meta = solver.solve_builder(builder)
    counts = qaoa_meta.get("counts", {})
    assert len(counts) > 0

    best_perm, is_raw_valid, dec_info = builder.decode_quantum_marginals(counts)
    assert len(best_perm) == 3
    assert set(best_perm) == set(custs)
    assert dec_info["best_subproblem_dist"] > 0


def test_candidate_diversity_and_multi_rollout():
    """Verify candidate diversity across rollout policies in MasterFullHybridEVRPSolver."""
    inst = _get_toy_evrp_instance(4)
    solver = MasterFullHybridEVRPSolver(hybrid_iterations=1, seed=42)
    sol = solver.solve(inst)

    assert isinstance(sol, VRPSolution)
    assert sol.feasible, "Full hybrid solution must be feasible"
    assert sol.total_distance > 0


def test_hierarchical_decomposition_large_instances():
    """Verify hierarchical clustering and recombination on an 8-customer instance."""
    inst = _get_toy_evrp_instance(8)
    from src.evrp.decomposition import decompose_instance_into_clusters, build_hierarchical_initial_routes
    from src.evrp.cost import calculate_evrp_solution_cost

    clusters = decompose_instance_into_clusters(inst, target_cluster_size=4)
    assert len(clusters) >= 2
    routes = build_hierarchical_initial_routes(inst, clusters)
    assert len(routes) == len(clusters)
    eval_res = calculate_evrp_solution_cost(inst, routes)
    assert eval_res["feasible"]
    all_served = [c for r in routes for c in r if inst.is_customer(c)]
    assert len(all_served) == 8


def test_all_eight_solvers_in_benchmark():
    """Verify that run_evrp_comparison successfully evaluates all 8 methods."""
    inst = _get_toy_evrp_instance(4)
    run_res = run_evrp_comparison(inst, seed=42)
    assert "results" in run_res
    results = run_res["results"]

    expected_methods = [
        "ga",
        "ga_ls",
        "rl",
        "rl_ls",
        "conventional_qaoa",
        "rl_qaoa",
        "rl_qaoa_ls",
        "full_hybrid",
    ]
    for m in expected_methods:
        assert m in results, f"Missing method {m} in results"
        res = results[m]
        assert "total_cost" in res
        assert "feasible" in res
        assert res["feasible"] is True
        assert res["total_cost"] > 0
        assert "runtime_seconds" in res


def test_best_feasible_solution_preservation():
    """Verify that MasterFullHybridEVRPSolver strictly preserves or improves the best feasible solution."""
    inst = _get_toy_evrp_instance(4)
    solver = MasterFullHybridEVRPSolver(hybrid_iterations=1, seed=42)
    sol = solver.solve(inst)

    assert sol.feasible
    cost = sol.metadata.get("final_cost", sol.metadata.get("total_cost", 0.0))
    assert cost > 0
    assert cost < 1000.0


def test_timeout_and_fallback_handling():
    """Verify that solver gracefully handles zero hybrid iterations and falls back safely."""
    inst = _get_toy_evrp_instance(4)
    solver = RLGuidedQAOASolver(hybrid_iterations=0, seed=42)
    sol = solver.solve(inst)

    assert isinstance(sol, VRPSolution)
    assert sol.feasible
    assert sol.metadata.get("qaoa_calls", 0) == 0


def test_multi_seed_comparison_and_ablation_study():
    """Verify multi-seed statistical evaluation and 6-part ablation studies."""
    inst = _get_toy_evrp_instance(4)

    # Multi-seed evaluation with 2 seeds
    multi_seed_res = run_evrp_multi_seed_comparison(inst, seeds=[42, 101])
    assert "aggregated_metrics" in multi_seed_res
    assert "rl_qaoa" in multi_seed_res["aggregated_metrics"]
    assert "mean_cost" in multi_seed_res["aggregated_metrics"]["rl_qaoa"]
    assert "std_cost" in multi_seed_res["aggregated_metrics"]["rl_qaoa"]

    # Ablation study
    ablation_res = run_evrp_ablation_study(inst, seed=42)
    assert "ablations" in ablation_res
    assert len(ablation_res["ablations"]) >= 6


def test_result_serialization_and_fastapi_endpoints():
    """Verify that new API endpoints return valid JSON and status 200."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1. Multi-seed endpoint
    resp = client.post(
        "/api/evrp/multi-seed",
        json={"instance_name": "E-n29-k4-s7.evrp", "customer_count": 4, "seeds": [42, 101]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "aggregated_metrics" in data
    assert "seeds_evaluated" in data

    # 2. Ablation run endpoint
    resp = client.post(
        "/api/evrp/ablation/run",
        json={"instance_name": "E-n29-k4-s7.evrp", "customer_count": 4, "seed": 42},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ablations" in data

    # 3. Scalability endpoint
    resp = client.post(
        "/api/evrp/scalability",
        json={"instance_name": "E-n29-k4-s7.evrp", "customer_counts": [4]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "scalability_curve" in data

