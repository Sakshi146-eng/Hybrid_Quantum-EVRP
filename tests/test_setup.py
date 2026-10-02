"""
Initial environment and package setup unit tests.
Verifies module imports, configuration loading, data generation, and basic EVRP feasibility checking.
"""
import math
import numpy as np

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.loader import DataLoader
from src.vrp.instance import VRPInstance
from src.evrp.instance import EVRPInstance
from src.evrp.feasibility import EVRPFeasibilityChecker
from src.evaluation.metrics import EvaluationMetrics


def test_config_loader():
    """Verify that default YAML configuration loads without errors."""
    config = load_config("configs/default_config.yaml")
    assert isinstance(config, dict)
    assert "project" in config
    assert config["project"]["name"] == "quantum-evrp"


def test_seed_reproducibility():
    """Verify random seed setting behavior."""
    set_seed(42)
    val1 = np.random.rand()
    set_seed(42)
    val2 = np.random.rand()
    assert val1 == val2


def test_data_loader_toy_instance():
    """Verify generation of toy VRP instance."""
    loader = DataLoader()
    instance_data = loader.create_toy_vrp_instance(num_nodes=4, seed=42)
    assert instance_data["num_nodes"] == 4
    assert instance_data["distance_matrix"].shape == (4, 4)
    # Check distance matrix symmetry and zero diagonal
    np.testing.assert_array_almost_equal(np.diag(instance_data["distance_matrix"]), 0.0)


def test_vrp_instance_cost():
    """Verify distance calculation in VRPInstance."""
    dist_matrix = np.array([
        [0.0, 3.0, 4.0],
        [3.0, 0.0, 5.0],
        [4.0, 5.0, 0.0]
    ])
    vrp = VRPInstance(distance_matrix=dist_matrix)
    route = [0, 1, 2, 0]
    # Cost = 0->1 (3.0) + 1->2 (5.0) + 2->0 (4.0) = 12.0
    cost = vrp.calculate_route_distance(route)
    assert cost == 12.0


def test_evrp_feasibility_checker():
    """Verify independent EVRP feasibility checker logic."""
    dist_matrix = np.array([
        [0.0, 5.0, 5.0],
        [5.0, 0.0, 2.0],
        [5.0, 2.0, 0.0]
    ])
    # Battery cap 10, consumption 1.0 per unit distance
    evrp = EVRPInstance(
        distance_matrix=dist_matrix,
        battery_capacity=10.0,
        energy_consumption_rate=1.0,
        charging_stations=[1],
        depot_idx=0,
        demands=np.array([0, 2, 3]),
        capacities=[10.0]
    )

    checker = EVRPFeasibilityChecker(evrp)
    # Route: 0 -> 1 -> 2 -> 0. Distances: 5 + 2 + 5 = 12.
    # At 1 (charging station), battery recharges to 10.
    # 0->1: energy 5, battery left 5. Recharge at 1 -> battery 10.
    # 1->2: energy 2, battery left 8.
    # 2->0: energy 5, battery left 3.
    # Feasible!
    feasible, details = checker.check_route_feasibility([0, 1, 2, 0])
    assert details["depot_start_end"] is True
    assert details["capacity"] is True
    assert details["battery"] is True
    assert feasible is True

    # Test battery infeasible route: 0 -> 2 -> 0 without charging (Distance: 5 + 5 = 10 energy, battery cap 4)
    evrp_tight = EVRPInstance(
        distance_matrix=dist_matrix,
        battery_capacity=4.0,
        energy_consumption_rate=1.0,
        charging_stations=[],
        depot_idx=0,
    )
    checker_tight = EVRPFeasibilityChecker(evrp_tight)
    feasible_tight, details_tight = checker_tight.check_route_feasibility([0, 2, 0])
    assert details_tight["battery"] is False
    assert feasible_tight is False


def test_evaluation_metrics():
    """Verify summary metrics calculation."""
    summary = EvaluationMetrics.compute_summary(
        total_distances=[10.0, 12.0, 14.0],
        feasibility_flags=[True, True, False],
        runtimes=[1.0, 1.5, 2.0]
    )
    assert math.isclose(summary["feasibility_rate"], 2.0 / 3.0, rel_tol=1e-2)
    assert summary["mean_total_distance"] == 12.0
    assert summary["min_total_distance"] == 10.0
