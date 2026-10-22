"""
Unit tests for Genetic Algorithm (GA) Classical VRP Solver (Phase 2).
Verifies solver instantiation, VRPSolution output, node coverage, permutation validity,
depot start/end checks, capacity bounds, seed reproducibility, and toy instance optimization.
"""
import math
from src.data.loader import DataLoader
from src.vrp.instance import VRPInstance
from src.classical.solution import VRPSolution
from src.classical.genetic_algorithm import GeneticAlgorithmSolver


def test_ga_solver_instantiation_and_name():
    """Verify GA solver can be instantiated and returns correct algorithm name."""
    solver = GeneticAlgorithmSolver(population_size=20, generations=10, seed=42)
    assert solver.algorithm_name == "Genetic Algorithm"
    assert solver.config.population_size == 20
    assert solver.config.generations == 10


def test_ga_solver_returns_vrp_solution():
    """Verify GA solver returns a VRPSolution object with populated attributes."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)

    solver = GeneticAlgorithmSolver(population_size=20, generations=20, seed=42)
    sol = solver.solve(inst)

    assert isinstance(sol, VRPSolution)
    assert sol.algorithm_name == "Genetic Algorithm"
    assert sol.runtime_seconds > 0.0
    assert isinstance(sol.routes, list)
    assert len(sol.routes) >= 1
    assert "best_distance_history" in sol.metadata


def test_ga_customer_coverage_and_no_duplicates():
    """Verify all customers are served exactly once across generated vehicle routes without duplicates."""
    loader = DataLoader()
    for n_cust in [3, 4, 5]:
        inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=n_cust)
        solver = GeneticAlgorithmSolver(population_size=30, generations=30, seed=42)
        sol = solver.solve(inst)

        # Collect all customer nodes visited in routes
        visited = []
        for route in sol.routes:
            cust_in_route = [node for node in route if node != inst.depot_idx]
            visited.extend(cust_in_route)

        # All customer nodes must be covered
        assert sorted(visited) == sorted(inst.customer_indices)
        # No duplicates
        assert len(visited) == len(set(visited))


def test_ga_routes_start_and_end_at_depot():
    """Verify all routes start and end at the depot index."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=4)
    solver = GeneticAlgorithmSolver(population_size=25, generations=25, seed=42)
    sol = solver.solve(inst)

    for route in sol.routes:
        assert len(route) >= 2
        assert route[0] == inst.depot_idx
        assert route[-1] == inst.depot_idx


def test_ga_vehicle_capacity_enforcement():
    """Verify payload capacity constraint is respected for feasible instances."""
    loader = DataLoader()
    # 3 customer instance: Demands 2.0, 3.0, 4.0. Capacity = 10.0
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)

    solver = GeneticAlgorithmSolver(population_size=30, generations=30, seed=42)
    sol = solver.solve(inst)

    assert sol.feasible is True
    for route, load in zip(sol.routes, sol.vehicle_loads):
        assert load <= inst.vehicle_capacity


def test_ga_total_distance_validity():
    """Verify returned total distance matches calculation from instance cost function."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=4)

    solver = GeneticAlgorithmSolver(population_size=25, generations=25, seed=42)
    sol = solver.solve(inst)

    calc_dist = inst.calculate_total_cost(sol.routes)
    assert math.isclose(sol.total_distance, calc_dist, rel_tol=1e-5)


def test_ga_seed_reproducibility():
    """Verify same random seed produces identical routes and distance results."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=5)

    solver1 = GeneticAlgorithmSolver(population_size=30, generations=40, seed=123)
    sol1 = solver1.solve(inst)

    solver2 = GeneticAlgorithmSolver(population_size=30, generations=40, seed=123)
    sol2 = solver2.solve(inst)

    assert sol1.routes == sol2.routes
    assert math.isclose(sol1.total_distance, sol2.total_distance, rel_tol=1e-6)


def test_ga_on_toy_instances():
    """Verify GA solves 3, 4, and 5 customer toy instances cleanly."""
    loader = DataLoader()
    for n_cust in [3, 4, 5]:
        inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=n_cust)
        solver = GeneticAlgorithmSolver(population_size=40, generations=50, seed=42)
        sol = solver.solve(inst)

        assert sol.feasible is True
        assert sol.total_distance > 0.0
        assert len(sol.routes) >= 1
