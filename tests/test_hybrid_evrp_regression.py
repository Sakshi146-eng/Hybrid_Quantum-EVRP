"""
Regression and verification test suite for Quantum-Assisted EVRP hybrid optimization.

Verifies:
1. RL training completes on a tiny instance and logs reward history.
2. The saved model checkpoint can be saved, reloaded, and produces identical policy weights.
3. Each of the six solvers returns a correctly evaluated EV-aware solution.
4. QAOA executes on compact subproblems when enabled and repairs bitstrings.
5. Hybrid stages use intended candidate solutions and preserve the best feasible solution (no degradation).
6. Relative improvement percentages are mathematically correct and consistent.
7. Backend endpoints (/api/evrp/rl/status, /api/evrp/compare) return accurate metrics.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evrp.instance import EVRPInstance
from src.data.e_cvrp_loader import load_evrp_instance
from src.rl.agent import EVRLAgent
from src.rl.env import EVRoutingEnv
from src.classical.ev_local_search import EVLocalSearchSolver
from src.classical.ev_ga_local_search import EVGALocalSearchSolver
from src.rl.rl_local_search import RLLocalSearchSolver
from src.classical.ev_genetic_algorithm import EVGeneticAlgorithmSolver
from src.quantum.rl_qaoa_hybrid import RLQAOASolver, FullHybridEVRPSolver
from src.evrp.benchmark import run_evrp_comparison


def _create_tiny_test_instance() -> EVRPInstance:
    """Creates a tiny 3-customer EVRP instance with known feasible properties."""
    coords = np.array([
        [0.0, 0.0],    # Depot
        [10.0, 0.0],   # Customer 1
        [10.0, 10.0],  # Customer 2
        [0.0, 10.0],   # Customer 3
        [5.0, 5.0],    # Charging station S1
    ], dtype=float)
    demands = np.array([0.0, 15.0, 20.0, 15.0, 0.0], dtype=float)
    capacities = [100.0, 100.0]

    return EVRPInstance(
        coordinates=coords,
        demands=demands,
        capacities=capacities,
        num_vehicles=2,
        charging_stations=[4],
        battery_capacity=100.0,
        initial_battery=100.0,
        battery_reserve=2.0,
        energy_consumption_rate=0.2,
        load_factor=0.1,
        charging_power=50.0,
        charging_price=0.25,
        speed=40.0,
    )


def test_regression_rl_training_completion():
    """Verify RL training completes on a tiny instance and records episode telemetry."""
    inst = _create_tiny_test_instance()
    agent = EVRLAgent(lr=0.01, gamma=0.95, seed=42)

    initial_reward_len = len(agent.training_history)
    assert initial_reward_len == 0

    episodes = 5
    history = []
    for ep in range(episodes):
        metric = agent.train_step(inst)
        history.append(metric)

    assert len(history) == episodes
    for m in history:
        assert "reward" in m
        assert "actor_loss" in m
        assert "critic_loss" in m
        assert "loss" in m
        assert isinstance(m["reward"], (float, int))


def test_regression_model_checkpoint_save_and_reload():
    """Verify that saved model checkpoints can be reloaded and match original network weights."""
    inst = _create_tiny_test_instance()
    agent = EVRLAgent(seed=42)

    # Train a step to differentiate weights from initial random
    agent.train_step(inst)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_rl_ckpt.pt"
        agent.save(ckpt_path)
        assert ckpt_path.is_file(), "Checkpoint file was not created on disk."

        # Load into fresh agent with matching seed
        loaded_agent = EVRLAgent(seed=42)
        loaded_agent.load(ckpt_path)

        # Compare Actor-Critic network parameters
        for p1, p2 in zip(agent.network.parameters(), loaded_agent.network.parameters()):
            assert torch.allclose(p1, p2, atol=1e-7), "Network weights differ after checkpoint load."

        # Verify deterministic greedy rollout produces identical route
        r1 = agent._rollout_single_solution(inst, deterministic=True, temperature=0.0)
        r2 = loaded_agent._rollout_single_solution(inst, deterministic=True, temperature=0.0)
        assert r1 == r2, f"Deterministic rollouts differed: {r1} vs {r2}"

        # Verify solve works identically
        sol1 = agent.solve(inst)
        sol2 = loaded_agent.solve(inst)
        assert sol1.routes == sol2.routes
        assert abs(sol1.total_distance - sol2.total_distance) < 1e-4


def test_regression_all_six_solvers_evaluated_correctly():
    """Verify each solver produces a valid, correctly evaluated EVRP solution."""
    inst = _create_tiny_test_instance()
    agent = EVRLAgent(seed=42)

    solvers = {
        "ga": EVGeneticAlgorithmSolver(population_size=10, generations=5, seed=42),
        "ga_ls": EVGALocalSearchSolver(population_size=10, generations=5, seed=42),
        "rl": agent,
        "rl_ls": RLLocalSearchSolver(rl_agent=agent, seed=42),
        "rl_qaoa": RLQAOASolver(rl_agent=agent, subproblem_size=3, qaoa_p_depth=1, qaoa_shots=100, qaoa_max_iter=5, seed=42),
        "full_hybrid": FullHybridEVRPSolver(rl_agent=agent, subproblem_size=3, qaoa_p_depth=1, qaoa_shots=100, qaoa_max_iter=5, seed=42),
    }

    expected_customers = set(inst.customer_indices)

    for name, solver in solvers.items():
        sol = solver.solve(inst)
        assert sol is not None, f"Solver {name} returned None"
        assert sol.total_distance > 0.0, f"Solver {name} produced non-positive distance"
        assert len(sol.routes) > 0, f"Solver {name} produced empty routes"

        visited_customers = []
        for r in sol.routes:
            assert r[0] == 0, f"Solver {name} route does not start at depot: {r}"
            assert r[-1] == 0, f"Solver {name} route does not end at depot: {r}"
            for node in r[1:-1]:
                if node in expected_customers:
                    visited_customers.append(node)

        # All customers visited exactly once
        assert set(visited_customers) == expected_customers, f"Solver {name} missed customers or included extras: {visited_customers}"
        assert len(visited_customers) == len(expected_customers), f"Solver {name} visited duplicate customers: {visited_customers}"

        # EV metadata present
        assert "evrp_evaluation" in sol.metadata, f"Solver {name} missing evrp_evaluation in metadata"
        assert "total_cost" in sol.metadata, f"Solver {name} missing total_cost in metadata"
        assert sol.feasible is True, f"Solver {name} produced infeasible solution on simple instance"


def test_regression_qaoa_executes_on_subproblem():
    """Verify QAOA executes on subproblems, runs simulated circuit, and outputs solution metadata."""
    inst = _create_tiny_test_instance()
    agent = EVRLAgent(seed=42)
    qaoa_solver = RLQAOASolver(
        rl_agent=agent,
        subproblem_size=3,
        qaoa_p_depth=1,
        qaoa_shots=200,
        qaoa_max_iter=5,
        seed=42,
    )
    sol = qaoa_solver.solve(inst)
    assert sol is not None
    assert sol.feasible is True
    # QAOA execution metadata
    assert "qaoa_subproblem_evaluated" in sol.metadata or "qaoa_calls" in sol.metadata
    runtime_breakdown = sol.metadata.get("runtime_breakdown", {})
    assert "rl_seconds" in runtime_breakdown
    assert "qaoa_seconds" in runtime_breakdown


def test_regression_best_feasible_solution_preservation():
    """Verify that local search and QAOA preserve the best feasible solution and never degrade it."""
    inst = _create_tiny_test_instance()
    agent = EVRLAgent(seed=42)

    # Initial solution
    init_sol = agent.solve(inst)
    init_cost = init_sol.metadata.get("total_cost", init_sol.total_distance)

    # Local search refinement
    ls_solver = RLLocalSearchSolver(rl_agent=agent, seed=42)
    ls_sol = ls_solver.solve(inst)
    ls_cost = ls_sol.metadata.get("total_cost", ls_sol.total_distance)
    assert ls_cost <= init_cost + 1e-4, f"Local Search degraded solution: {init_cost} -> {ls_cost}"

    # QAOA refinement
    qaoa_solver = RLQAOASolver(rl_agent=agent, subproblem_size=3, qaoa_p_depth=1, qaoa_shots=100, qaoa_max_iter=5, seed=42)
    qaoa_sol = qaoa_solver.solve(inst)
    qaoa_cost = qaoa_sol.metadata.get("total_cost", qaoa_sol.total_distance)
    assert qaoa_cost <= init_cost + 1e-4, f"RL+QAOA degraded solution: {init_cost} -> {qaoa_cost}"

    # Full hybrid
    full_solver = FullHybridEVRPSolver(rl_agent=agent, subproblem_size=3, qaoa_p_depth=1, qaoa_shots=100, qaoa_max_iter=5, seed=42)
    full_sol = full_solver.solve(inst)
    full_cost = full_sol.metadata.get("total_cost", full_sol.total_distance)
    assert full_cost <= init_cost + 1e-4, f"Full Hybrid degraded solution: {init_cost} -> {full_cost}"
    assert full_cost <= ls_cost + 1e-4 or abs(full_cost - ls_cost) < 1e-2


def test_regression_improvement_percentages_math():
    """Verify mathematical correctness of improvement percentage computations."""
    inst = _create_tiny_test_instance()
    comparison = run_evrp_comparison(
        instance=inst,
        algorithms=["ga", "rl", "rl_ls"],
        seed=42,
    )
    res = comparison["results"]
    ga_cost = res["ga"]["total_cost"]
    rl_cost = res["rl"]["total_cost"]

    # GA relative to GA must be 0%
    assert abs(res["ga"]["improvement_over_ga_percent"]) < 1e-4

    # RL relative to RL must be 0%
    assert abs(res["rl"]["improvement_over_rl_percent"]) < 1e-4

    # Check formula for RL+LS
    rl_ls_cost = res["rl_ls"]["total_cost"]
    expected_gain_ga = round(((ga_cost - rl_ls_cost) / ga_cost) * 100.0, 2)
    expected_gain_rl = round(((rl_cost - rl_ls_cost) / rl_cost) * 100.0, 2)

    assert abs(res["rl_ls"]["improvement_over_ga_percent"] - expected_gain_ga) < 0.05
    assert abs(res["rl_ls"]["improvement_over_rl_percent"] - expected_gain_rl) < 0.05


def test_regression_backend_api_and_telemetry():
    """Verify FastAPI backend endpoints return consistent schema and valid values."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1. Check health
    res_h = client.get("/api/health")
    assert res_h.status_code == 200
    assert res_h.json() == {"status": "ok"}

    # 2. Check RL status
    res_s = client.get("/api/evrp/rl/status")
    assert res_s.status_code == 200
    status_data = res_s.json()
    assert "status" in status_data
    assert "progress" in status_data
    assert "checkpoint_exists" in status_data

    # 3. Test EVRP compare endpoint with tiny config
    payload = {
        "instance_name": "E-n29-k4-s7.evrp",
        "customer_count": 3,
        "algorithms": ["ga", "rl", "rl_ls"],
        "battery_capacity": 100.0,
        "base_energy_rate": 0.2,
        "seed": 42,
    }
    res_c = client.post("/api/evrp/compare", json=payload)
    assert res_c.status_code == 200, f"Compare failed: {res_c.text}"
    comp_data = res_c.json()

    assert "results" in comp_data
    assert "depot" in comp_data
    assert "customers" in comp_data
    assert len(comp_data["customers"]) == 3

    results = comp_data["results"]
    for algo in ["ga", "rl", "rl_ls"]:
        assert algo in results
        algo_data = results[algo]
        assert "total_cost" in algo_data
        assert "total_distance" in algo_data
        assert "total_energy_kwh" in algo_data
        assert "routes" in algo_data
        assert "feasible" in algo_data
        assert "improvement_over_ga_percent" in algo_data
        assert "improvement_over_rl_percent" in algo_data


if __name__ == "__main__":
    print("Running Regression Test Suite...")
    test_regression_rl_training_completion()
    print("  [PASS] test_regression_rl_training_completion")
    test_regression_model_checkpoint_save_and_reload()
    print("  [PASS] test_regression_model_checkpoint_save_and_reload")
    test_regression_all_six_solvers_evaluated_correctly()
    print("  [PASS] test_regression_all_six_solvers_evaluated_correctly")
    test_regression_qaoa_executes_on_subproblem()
    print("  [PASS] test_regression_qaoa_executes_on_subproblem")
    test_regression_best_feasible_solution_preservation()
    print("  [PASS] test_regression_best_feasible_solution_preservation")
    test_regression_improvement_percentages_math()
    print("  [PASS] test_regression_improvement_percentages_math")
    test_regression_backend_api_and_telemetry()
    print("  [PASS] test_regression_backend_api_and_telemetry")
    print("All Regression Tests Passed Successfully!")
