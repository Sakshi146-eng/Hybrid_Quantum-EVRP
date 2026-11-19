"""
Unit tests for Classical VRP Baseline Base Solver & Solution Architecture (Phase 1).
Verifies VRPSolution data fields, dictionary conversion, abstract BaseVRPSolver behavior,
runtime tracking, route cost evaluation, vehicle load calculation, and feasibility checks.
"""
import math
from src.data.loader import DataLoader
from src.vrp.instance import VRPInstance
from src.classical.solution import VRPSolution
from src.classical.base_solver import BaseVRPSolver, RandomBaselineSolver


def test_vrp_solution_structure_and_dict():
    """Verify VRPSolution attributes, serialization to dict, and summary string output."""
    sol = VRPSolution(
        algorithm_name="Test Solver",
        routes=[[0, 1, 3, 0], [0, 2, 0]],
        total_distance=15.5,
        feasible=True,
        runtime_seconds=0.012345,
        vehicle_loads=[6.0, 3.0],
        metadata={"pop_size": 50}
    )

    assert sol.algorithm_name == "Test Solver"
    assert sol.routes == [[0, 1, 3, 0], [0, 2, 0]]
    assert math.isclose(sol.total_distance, 15.5)
    assert sol.feasible is True
    assert math.isclose(sol.runtime_seconds, 0.012345)
    assert sol.vehicle_loads == [6.0, 3.0]
    assert sol.metadata["pop_size"] == 50

    # Test dictionary conversion for frontend JSON API
    d = sol.to_dict()
    assert isinstance(d, dict)
    assert d["algorithm_name"] == "Test Solver"
    assert d["routes"] == [[0, 1, 3, 0], [0, 2, 0]]
    assert d["total_distance"] == 15.5
    assert d["feasible"] is True
    assert d["vehicle_loads"] == [6.0, 3.0]

    # Test summary string
    summary = sol.summary()
    assert "Algorithm      : Test Solver" in summary
    assert "Feasible       : YES" in summary
    assert "Vehicle 1: 0 -> 1 -> 3 -> 0 (Load: 6.0)" in summary


def test_abstract_base_solver_enforcement():
    """Verify that BaseVRPSolver cannot be instantiated directly without implementing abstract methods."""
    try:
        solver = BaseVRPSolver()
        assert False, "Expected TypeError when instantiating abstract BaseVRPSolver"
    except TypeError:
        pass


def test_random_baseline_solver_execution():
    """Verify solver execution flow, runtime recording, distance cost calculation, and payload load check."""
    loader = DataLoader()
    # 3-customer toy instance: Depot=0 (0,0), C1 (0,3) d=2, C2 (4,0) d=3, C3 (4,3) d=4. Cap=10.0
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)

    solver = RandomBaselineSolver(seed=42)
    solution = solver.solve(inst)

    assert isinstance(solution, VRPSolution)
    assert solution.algorithm_name == "Random Baseline"
    assert len(solution.routes) == 1
    # Route: [0, 1, 2, 3, 0]
    assert solution.routes[0] == [0, 1, 2, 3, 0]
    # Distance: dist(0,1)=3 + dist(1,2)=5 + dist(2,3)=3 + dist(3,0)=5 = 16.0
    assert math.isclose(solution.total_distance, 16.0, rel_tol=1e-5)
    # Total demand = 2 + 3 + 4 = 9.0 <= 10.0 -> Feasible
    assert solution.feasible is True
    assert solution.vehicle_loads == [9.0]
    assert solution.runtime_seconds >= 0.0


def test_solver_feasibility_check_infeasible_capacity():
    """Verify that VRPSolution correctly flags feasibility = False if capacity is exceeded."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)

    # Tight capacity = 5.0 (visited demand 9.0 > 5.0)
    inst_tight = VRPInstance(
        distance_matrix=inst.distance_matrix,
        coordinates=inst.coordinates,
        demands=inst.demands,
        vehicle_capacity=5.0
    )

    solver = RandomBaselineSolver(seed=42)
    solution = solver.solve(inst_tight)

    assert solution.feasible is False
    assert solution.vehicle_loads == [9.0]
