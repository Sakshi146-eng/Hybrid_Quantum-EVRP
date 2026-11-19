"""
Comprehensive automated tests for EVRP Model, Reinforcement Learning, QAOA, and Hybrid Solvers.
"""
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from src.evrp.instance import EVRPInstance
from src.evrp.feasibility import EVRPFeasibilityChecker
from src.evrp.charging import (
    find_best_charging_station,
    insert_charging_stations_if_needed,
    prune_redundant_charging_stops,
)
from src.evrp.cost import calculate_evrp_solution_cost
from src.classical.ev_genetic_algorithm import EVGeneticAlgorithmSolver
from src.classical.ev_ga_local_search import EVGALocalSearchSolver
from src.rl.env import EVRoutingEnv
from src.rl.agent import EVRLAgent
from src.quantum.rl_qaoa_hybrid import RLQAOASolver, FullHybridEVRPSolver
from src.data.e_cvrp_loader import load_evrp_instance
from backend.main import app

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "e_cvrp" / "E-n29-k4-s7.evrp"


def _create_small_test_instance():
    """Constructs a deterministic 3-customer, 1-charging-station EVRPInstance."""
    coords = np.array([
        [0.0, 0.0],    # 0: Depot
        [10.0, 0.0],   # 1: Cust 1
        [10.0, 10.0],  # 2: Cust 2
        [0.0, 10.0],   # 3: Cust 3
        [5.0, 5.0],    # 4: Charging station
    ], dtype=float)
    demands = np.array([0.0, 5.0, 5.0, 5.0, 0.0], dtype=float)
    capacities = [20.0, 20.0, 20.0]

    return EVRPInstance(
        coordinates=coords,
        demands=demands,
        capacities=capacities,
        num_vehicles=3,
        charging_stations=[4],
        battery_capacity=25.0,
        initial_battery=25.0,
        battery_reserve=1.0,
        energy_consumption_rate=0.2,   # 0.2 kWh/km
        load_factor=0.1,
        charging_power=50.0,
        charging_price=0.25,
        speed=40.0,
    )


def test_ev_energy_consumption_and_load_factor():
    """Validates distance and load-dependent energy calculation formula."""
    inst = _create_small_test_instance()
    # Leg from 0 to 1: distance = 10 km, load = 15.0, Q = 20.0
    # Energy = 10 * 0.2 * (1.0 + 0.1 * (15/20)) = 10 * 0.2 * (1.075) = 2.15 kWh
    e = inst.calculate_leg_energy(0, 1, current_load=15.0)
    assert abs(e - 2.15) < 1e-4, f"Expected 2.15 kWh, got {e}"


def test_ev_leg_by_leg_battery_tracking():
    """Validates leg-by-leg state-of-charge calculation and violation detection."""
    inst = _create_small_test_instance()
    eval_res = inst.evaluate_route_ev([0, 1, 2, 3, 0])
    assert eval_res["feasible"], f"Expected route to be feasible, got {eval_res['violations']}"
    assert len(eval_res["leg_trace"]) == 4

    # If battery capacity is lowered to 5.0, the route must be infeasible due to battery depletion
    inst_small = _create_small_test_instance()
    inst_small.battery_capacity = 5.0
    inst_small.initial_battery = 5.0
    eval_res_small = inst_small.evaluate_route_ev([0, 1, 2, 3, 0])
    assert not eval_res_small["feasible"], "Expected small battery route to be infeasible"
    assert any("Battery" in v for v in eval_res_small["violations"])


def test_ev_charging_station_insertion():
    """Validates dynamic insertion of charging stations into battery-depleted routes."""
    inst = _create_small_test_instance()
    inst.battery_capacity = 5.0
    inst.initial_battery = 5.0

    infeasible_route = [0, 1, 2, 3, 0]
    repaired, is_feasible = insert_charging_stations_if_needed(infeasible_route, inst)
    # The repaired route should contain the charging station (node 4)
    assert 4 in repaired, f"Expected charging station 4 in repaired route, got {repaired}"
    assert is_feasible is True
    eval_rep = inst.evaluate_route_ev(repaired)
    assert eval_rep["feasible"], f"Repaired route should be feasible: {eval_rep['violations']}"


def test_ev_charging_station_pruning():
    """Validates that redundant back-to-back charging stops are safely pruned."""
    inst = _create_small_test_instance()
    # Route with duplicate consecutive charging stations: [0, 4, 4, 1, 2, 3, 0]
    route_with_dups = [0, 4, 4, 1, 2, 3, 0]
    pruned = prune_redundant_charging_stops(route_with_dups, inst)
    assert pruned.count(4) <= 1, f"Redundant charging stops should be pruned: {pruned}"


def test_ev_feasibility_checker():
    """Validates EVRPFeasibilityChecker on multi-vehicle and single-route configurations."""
    inst = _create_small_test_instance()
    checker = EVRPFeasibilityChecker(inst)

    # Valid route
    is_valid, errs = checker.check_route_feasibility([0, 1, 2, 3, 0])
    assert is_valid, f"Expected valid route, got {errs}"

    # Invalid route: missing depot end
    is_valid_end, _ = checker.check_route_feasibility([0, 1, 2, 3])
    assert not is_valid_end, "Route without depot end should fail"

    # Multi-vehicle complete solution
    sol_valid, _ = checker.check_solution_feasibility([[0, 1, 2, 3, 0]])
    assert sol_valid, "Solution covering all customers should be valid"


def test_ev_cost_calculation():
    """Validates unified multi-objective EVRP cost formulation."""
    inst = _create_small_test_instance()
    details = calculate_evrp_solution_cost(
        instance=inst,
        routes=[[0, 1, 2, 3, 0]],
        w_dist=1.0,
        w_charging=1.0,
        w_vehicle=5.0,
    )
    assert details["feasible"] is True
    assert details["penalty_cost"] == 0.0
    assert details["total_cost"] > 0.0


def test_ev_genetic_algorithm_solver():
    """Validates that EVGeneticAlgorithmSolver produces 100% feasible EVRP solutions."""
    inst = _create_small_test_instance()
    solver = EVGeneticAlgorithmSolver(population_size=20, generations=15, seed=42)
    sol = solver.solve(inst)
    assert sol.feasible is True
    assert len(sol.routes) >= 1
    assert sol.total_distance > 0.0


def test_ev_ga_local_search_solver():
    """Validates that EVGALocalSearchSolver improves or maintains feasibility."""
    inst = _create_small_test_instance()
    solver = EVGALocalSearchSolver(population_size=20, generations=15, seed=42)
    sol = solver.solve(inst)
    assert sol.feasible is True
    assert sol.metadata.get("feasible_found", True) is True


def test_ev_routing_gym_env():
    """Validates Gymnasium EVRoutingEnv reset, step, observation spaces, and action masks."""
    inst = _create_small_test_instance()
    env = EVRoutingEnv(instance=inst)
    obs, info = env.reset()

    assert "action_mask" in info
    mask = info["action_mask"]
    assert len(mask) == env.action_space.n
    assert mask[1] == 1  # Customer 1 is a valid reachable action

    # Step to customer 1
    next_obs, reward, terminated, truncated, info = env.step(1)
    assert not np.isnan(reward)
    assert env.current_node == 1


def test_ev_rl_agent_train_and_inference():
    """Validates EVRLAgent training step and deterministic route generation."""
    inst = _create_small_test_instance()
    agent = EVRLAgent(seed=42)
    metrics = agent.train_step(inst)
    assert "reward" in metrics
    assert "loss" in metrics

    # Greedy inference
    sol = agent.solve(inst)
    assert sol.total_distance > 0.0
    assert len(sol.routes) >= 1


def test_rl_qaoa_hybrid_solver():
    """Validates RLQAOASolver window-based QAOA local optimization."""
    inst = _create_small_test_instance()
    agent = EVRLAgent(seed=42)
    solver = RLQAOASolver(
        rl_agent=agent,
        subproblem_size=3,
        qaoa_p_depth=1,
        qaoa_shots=300,
        qaoa_max_iter=10,
        hybrid_iterations=1,
        seed=42,
    )
    sol = solver.solve(inst)
    assert sol.feasible is True
    assert sol.total_distance > 0.0
    assert "qaoa_calls" in sol.metadata


def test_full_hybrid_evrp_solver():
    """Validates FullHybridEVRPSolver end-to-end execution."""
    inst = _create_small_test_instance()
    agent = EVRLAgent(seed=42)
    solver = FullHybridEVRPSolver(
        rl_agent=agent,
        subproblem_size=3,
        qaoa_p_depth=1,
        qaoa_shots=300,
        qaoa_max_iter=10,
        hybrid_iterations=1,
        seed=42,
    )
    sol = solver.solve(inst)
    assert sol.feasible is True
    assert "evrp_evaluation" in sol.metadata


def test_evrp_api_endpoints():
    """Validates FastAPI EVRP comparison, RL status, and ablation endpoints."""
    client = TestClient(app)

    # 1. RL Status
    status_res = client.get("/api/evrp/rl/status")
    assert status_res.status_code == 200
    assert "status" in status_res.json()

    # 2. EVRP Comparison endpoint
    if BENCHMARK_PATH.is_file():
        compare_payload = {
            "instance_name": "E-n29-k4-s7.evrp",
            "customer_count": 3,
            "algorithms": ["ga", "rl"],
            "seed": 42,
        }
        res = client.post("/api/evrp/compare", json=compare_payload)
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert "ga" in data["results"]
        assert "rl" in data["results"]
