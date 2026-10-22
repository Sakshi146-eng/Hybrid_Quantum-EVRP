"""
Unit tests for CEC-2020 E-CVRP Benchmark Dataset Loader and GA Integration.
"""
from pathlib import Path
import tempfile
import numpy as np

from src.data.e_cvrp_loader import parse_e_cvrp_file, load_e_cvrp_instance
from src.vrp.instance import VRPInstance
from src.vrp.validator import VRPValidator
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.classical.solution import VRPSolution
from src.evaluation.benchmark import record_benchmark_result, BenchmarkResult

BENCHMARK_PATH = Path("data/raw/e_cvrp/E-n29-k4-s7.evrp")


def _get_test_instance_path() -> Path:
    """Helper to get benchmark file path or create a mock .evrp temp file if missing."""
    if BENCHMARK_PATH.is_file():
        return BENCHMARK_PATH

    # Create temporary mock .evrp file
    temp_dir = Path(tempfile.gettempdir())
    mock_file = temp_dir / "mock_test.evrp"
    mock_content = """NAME: mock_test.evrp
TYPE: EVRP
OPTIMAL_VALUE: 100
VEHICLES: 2
DIMENSION: 6
STATIONS: 1
CAPACITY: 50.0
ENERGY_CAPACITY: 100.0
ENERGY_CONSUMPTION: 1.0
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 3.0 4.0
3 6.0 8.0
4 1.0 1.0
5 2.0 2.0
6 10.0 10.0
DEMAND_SECTION
1 0
2 10
3 20
4 15
5 5
STATIONS_COORD_SECTION
6
DEPOT_SECTION
1
-1
EOF
"""
    with open(mock_file, "w", encoding="utf-8") as f:
        f.write(mock_content)
    return mock_file


def test_benchmark_loader_valid_instance_loads():
    path = _get_test_instance_path()
    instance = load_e_cvrp_instance(path)
    assert isinstance(instance, VRPInstance), "Loaded object must be an instance of VRPInstance"
    assert hasattr(instance, "raw_data"), "VRPInstance must attach raw_data metadata dictionary"


def test_benchmark_depot_identification():
    path = _get_test_instance_path()
    instance = load_e_cvrp_instance(path)
    assert instance.depot_idx == 0, "Depot index in VRPInstance must be 0"
    assert instance.demands[instance.depot_idx] == 0.0, "Depot demand must be 0.0"


def test_benchmark_customer_count():
    path = _get_test_instance_path()
    instance = load_e_cvrp_instance(path)
    if path == BENCHMARK_PATH:
        # E-n29-k4-s7 has 29 total nodes, 1 depot, 7 stations -> 21 customers
        assert instance.num_customers == 21, f"Expected 21 customers, got {instance.num_customers}"
        assert instance.num_nodes == 22, f"Expected 22 total nodes (depot + 21 customers), got {instance.num_nodes}"
    else:
        # mock has 6 total nodes, 1 depot, 1 station -> 4 customers
        assert instance.num_customers == 4
        assert instance.num_nodes == 5


def test_benchmark_vehicle_count_and_capacity():
    path = _get_test_instance_path()
    instance = load_e_cvrp_instance(path)
    if path == BENCHMARK_PATH:
        assert instance.num_vehicles == 4, f"Expected 4 vehicles, got {instance.num_vehicles}"
        assert instance.vehicle_capacity == 6000.0, f"Expected capacity 6000.0, got {instance.vehicle_capacity}"
    else:
        assert instance.num_vehicles == 2
        assert instance.vehicle_capacity == 50.0


def test_benchmark_demands_and_coords_parsing():
    path = _get_test_instance_path()
    instance = load_e_cvrp_instance(path)

    assert len(instance.demands) == instance.num_nodes, "Demands array length must match num_nodes"
    assert instance.coordinates.shape == (instance.num_nodes, 2), "Coordinates shape must be (num_nodes, 2)"
    assert instance.distance_matrix.shape == (instance.num_nodes, instance.num_nodes), "Distance matrix shape mismatch"
    assert np.allclose(np.diag(instance.distance_matrix), 0.0), "Distance matrix diagonal must be 0.0"


def test_benchmark_instance_vrp_validation():
    path = _get_test_instance_path()
    instance = load_e_cvrp_instance(path)

    val_res = VRPValidator.validate_instance_data(
        num_nodes=instance.num_nodes,
        depot_idx=instance.depot_idx,
        distance_matrix=instance.distance_matrix,
        demands=instance.demands,
        vehicle_capacity=instance.vehicle_capacity
    )
    assert val_res["is_valid"], f"Loaded instance validation failed: {val_res['errors']}"


def test_benchmark_ga_execution_and_vrp_solution():
    path = _get_test_instance_path()
    instance = load_e_cvrp_instance(path)

    solver = GeneticAlgorithmSolver(
        population_size=30,
        generations=20,
        seed=42
    )

    solution = solver.solve(instance)
    assert isinstance(solution, VRPSolution), "GA must return a VRPSolution instance"
    assert solution.algorithm_name == "Genetic Algorithm"
    assert isinstance(solution.feasible, bool)
    assert solution.total_distance > 0.0, "Total distance must be positive"
    assert len(solution.routes) > 0, "Routes list must not be empty"

    for r in solution.routes:
        assert r[0] == instance.depot_idx, "Route must start at depot (0)"
        assert r[-1] == instance.depot_idx, "Route must end at depot (0)"


def test_benchmark_result_recording():
    path = _get_test_instance_path()
    instance = load_e_cvrp_instance(path)

    solver = GeneticAlgorithmSolver(
        population_size=20,
        generations=10,
        seed=42
    )
    solution = solver.solve(instance)

    res = record_benchmark_result(solution, instance, instance_name="test_inst")
    assert isinstance(res, BenchmarkResult)
    assert res.instance_name == "test_inst"
    assert res.algorithm == "Genetic Algorithm"
    assert res.num_customers == instance.num_customers
    assert res.num_vehicles == instance.num_vehicles
    assert res.vehicle_capacity == instance.vehicle_capacity
    assert res.total_distance == solution.total_distance
    assert res.feasible == solution.feasible
    assert res.runtime_seconds == solution.runtime_seconds

    res_dict = res.to_dict()
    assert "instance_name" in res_dict
    assert "vehicle_loads" in res_dict
