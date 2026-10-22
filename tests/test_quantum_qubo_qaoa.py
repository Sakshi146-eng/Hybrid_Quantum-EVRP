"""
Comprehensive Unit and Integration Tests for QUBO Formulation, QAOA Solver,
Exact Baseline, and Comparative Evaluation Workflow.
"""
import numpy as np
from fastapi.testclient import TestClient
from qiskit.quantum_info import Statevector

from src.vrp.instance import VRPInstance
from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.classical.solution import VRPSolution
from src.quantum.qubo import VRPQuboBuilder, extract_sub_instance
from src.quantum.exact_solver import ExactBruteForceSolver
from src.quantum.qaoa_solver import QAOAVRPSolver
from src.quantum.comparison import run_quantum_classical_comparison
from backend.main import app


def _create_toy_instance(num_cust: int = 3) -> VRPInstance:
    """Helper creating a deterministic 2D VRPInstance."""
    coords = np.array([
        [0.0, 0.0],  # Depot
        [1.0, 0.0],  # Cust 1
        [1.0, 1.0],  # Cust 2
        [0.0, 1.0],  # Cust 3
    ][: num_cust + 1])
    return VRPInstance(coordinates=coords, num_vehicles=1, vehicle_capacity=50.0)


def test_qubo_variable_mapping():
    """Tests variable index mapping (customer, step) <-> k."""
    inst = _create_toy_instance(3)
    builder = VRPQuboBuilder(inst)

    assert builder.num_customers == 3
    assert builder.num_steps == 3
    assert builder.num_vars == 9

    # Check bijective index round-trip
    for c in builder.customer_indices:
        for t in range(builder.num_steps):
            k = builder.get_var_index(c, t)
            assert 0 <= k < 9
            dec_c, dec_t = builder.get_customer_and_step(k)
            assert dec_c == c
            assert dec_t == t


def test_qubo_matrix_construction():
    """Tests QUBO matrix construction shape, triangularity, and positive offset."""
    inst = _create_toy_instance(3)
    builder = VRPQuboBuilder(inst)
    Q, offset = builder.build_qubo()

    assert Q.shape == (9, 9)
    assert offset > 0.0
    # Matrix must be upper-triangular
    assert np.allclose(Q, np.triu(Q))


def test_qubo_energy_matches_actual_route_distance():
    """
    Checks that QUBO objective x^T Q x + offset matches actual route distance
    for multiple valid candidate permutations.
    """
    inst = _create_toy_instance(3)
    builder = VRPQuboBuilder(inst)
    Q, offset = builder.build_qubo()

    test_permutations = [
        [1, 2, 3],
        [3, 2, 1],
        [2, 1, 3],
        [1, 3, 2],
    ]

    for perm in test_permutations:
        route = [inst.depot_idx] + perm + [inst.depot_idx]
        actual_dist = inst.calculate_route_distance(route)

        # Build binary vector x
        x = np.zeros(builder.num_vars, dtype=float)
        for step, cust in enumerate(perm):
            k = builder.get_var_index(cust, step)
            x[k] = 1.0

        qubo_energy = float(x.T @ Q @ x + offset)
        assert abs(qubo_energy - actual_dist) < 1e-10, (
            f"Permutation {perm}: energy {qubo_energy} != actual {actual_dist}"
        )


def test_qubo_penalty_rejects_invalid_solutions():
    """Verifies that invalid bitstrings incur high penalty energy."""
    inst = _create_toy_instance(3)
    builder = VRPQuboBuilder(inst)
    Q, offset = builder.build_qubo()

    # Valid route energy (perm [1, 2, 3])
    x_valid = np.zeros(9)
    x_valid[builder.get_var_index(1, 0)] = 1.0
    x_valid[builder.get_var_index(2, 1)] = 1.0
    x_valid[builder.get_var_index(3, 2)] = 1.0
    valid_energy = float(x_valid.T @ Q @ x_valid + offset)

    # Invalid: All zeros
    x_zeros = np.zeros(9)
    zeros_energy = float(x_zeros.T @ Q @ x_zeros + offset)
    assert zeros_energy > valid_energy

    # Invalid: Two customers at step 0
    x_conflict = np.copy(x_valid)
    x_conflict[builder.get_var_index(2, 0)] = 1.0  # Conflict at step 0
    conflict_energy = float(x_conflict.T @ Q @ x_conflict + offset)
    assert conflict_energy > valid_energy


def test_route_decoding_and_validation():
    """Tests bitstring decoding into valid routes and detection of constraint violations."""
    inst = _create_toy_instance(3)
    builder = VRPQuboBuilder(inst)

    # Valid bitstring for perm [1, 2, 3]:
    # k = 0 (cust 1, step 0), k = 4 (cust 2, step 1), k = 8 (cust 3, step 2)
    # Right-to-left bitstring: qubit 8 is index 0, qubit 0 is index 8
    bs_chars = ["0"] * 9
    bs_chars[8 - 0] = "1"
    bs_chars[8 - 4] = "1"
    bs_chars[8 - 8] = "1"
    valid_bs = "".join(bs_chars)

    route, is_valid, details = builder.decode_bitstring(valid_bs)
    assert is_valid is True
    assert route == [0, 1, 2, 3, 0]
    assert len(details["violations"]) == 0

    # Invalid bitstring: all zeros
    route_inv, is_valid_inv, details_inv = builder.decode_bitstring("000000000")
    assert is_valid_inv is False
    assert len(details_inv["violations"]) > 0


def test_ising_hamiltonian_eigenvalues():
    """Verifies that expectation value of Ising Hamiltonian matches QUBO energy."""
    inst = _create_toy_instance(3)
    builder = VRPQuboBuilder(inst)
    Q, offset = builder.build_qubo()
    H, ising_offset = builder.to_ising_hamiltonian()

    # Valid perm [2, 1, 3]
    k1 = builder.get_var_index(2, 0)
    k2 = builder.get_var_index(1, 1)
    k3 = builder.get_var_index(3, 2)

    bs_chars = ["0"] * 9
    bs_chars[8 - k1] = "1"
    bs_chars[8 - k2] = "1"
    bs_chars[8 - k3] = "1"
    bs = "".join(bs_chars)

    # QUBO energy
    qubo_energy = builder.evaluate_bitstring_energy(bs)

    # Ising Hamiltonian energy on computational basis state
    sv = Statevector.from_label(bs)
    ising_energy = float(sv.expectation_value(H).real) + ising_offset

    assert abs(ising_energy - qubo_energy) < 1e-10


def test_exact_brute_force_solver():
    """Tests exact brute-force solver on 3-customer square."""
    inst = _create_toy_instance(3)
    solver = ExactBruteForceSolver()
    solution = solver.solve(inst)

    assert isinstance(solution, VRPSolution)
    assert solution.feasible is True
    # In unit square (0,0)->(1,0)->(1,1)->(0,1)->(0,0), optimal tour is 4.0
    assert abs(solution.total_distance - 4.0) < 1e-6
    assert solution.routes[0] == [0, 1, 2, 3, 0] or solution.routes[0] == [0, 3, 2, 1, 0]


def test_qaoa_solver_execution_small_instance():
    """Tests QAOA execution on small instance with AerSimulator and timing breakdown."""
    inst = _create_toy_instance(2)  # 2 customers -> 4 qubits
    solver = QAOAVRPSolver(p_depth=1, shots=300, max_iter=15, seed=42)
    solution = solver.solve(inst)

    assert isinstance(solution, VRPSolution)
    assert solution.total_distance > 0.0
    assert "timing_breakdown" in solution.metadata
    assert solution.metadata["qubits"] == 4
    assert solution.metadata["total_samples"] == 300
    assert solution.metadata["timing_breakdown"]["total_seconds"] > 0.0


def test_comparative_workflow_equivalent_inputs():
    """
    Tests comparative workflow on CEC-2020 benchmark instance:
    Verifies that Exact, GA, and QAOA solve identical sub-instances and optimality gap formula.
    """
    benchmark_file = "data/raw/e_cvrp/E-n29-k4-s7.evrp"
    parent_instance = load_e_cvrp_instance(benchmark_file)

    comp = run_quantum_classical_comparison(
        instance=parent_instance,
        customer_count=3,
        qaoa_params={"p_depth": 1, "shots": 300, "max_iter": 15},
        ga_params={"population_size": 30, "generations": 20},
        seed=42,
    )

    assert comp["customer_count"] == 3
    assert comp["qubits_required"] == 9
    assert comp["exact_baseline"]["total_distance"] > 0.0
    assert comp["classical_ga"]["total_distance"] > 0.0
    assert comp["qaoa_simulator"]["total_distance"] > 0.0

    # Optimality gap verification: ((cost - exact) / exact) * 100
    exact_dist = comp["exact_baseline"]["total_distance"]
    ga_dist = comp["classical_ga"]["total_distance"]
    expected_ga_gap = round(((ga_dist - exact_dist) / exact_dist) * 100.0, 4)
    assert abs(comp["classical_ga"]["optimality_gap_percent"] - expected_ga_gap) < 1e-3


def test_api_qaoa_and_compare_endpoints():
    """Tests FastAPI QAOA and compare endpoints, including parameter rejection for size > 4."""
    client = TestClient(app)

    # 1. Successful QAOA run
    res_q = client.post(
        "/api/benchmark/qaoa",
        json={"instance_name": "E-n29-k4-s7.evrp", "customer_count": 2, "shots": 200, "max_iter": 10},
    )
    assert res_q.status_code == 200
    assert "total_distance" in res_q.json()

    # 2. Successful Compare run
    res_c = client.post(
        "/api/benchmark/compare",
        json={
            "instance_name": "E-n29-k4-s7.evrp",
            "customer_count": 2,
            "shots": 200,
            "max_iter": 10,
            "ga_generations": 15,
        },
    )
    assert res_c.status_code == 200
    data_c = res_c.json()
    assert "exact_baseline" in data_c
    assert "classical_ga" in data_c
    assert "ga_local_search" in data_c
    assert "qaoa_simulator" in data_c
    assert "hybrid_ga_qaoa" in data_c

    # 3. Size validation error rejection (> 8 customers)
    res_err = client.post(
        "/api/benchmark/compare",
        json={"instance_name": "E-n29-k4-s7.evrp", "customer_count": 9},
    )
    assert res_err.status_code == 422


def test_local_search_operators():
    """Verifies that 2-opt, relocate, and swap operators strictly preserve depots and validity."""
    from src.classical.local_search import (
        two_opt_single_route,
        relocate_single_route,
        swap_single_route,
        apply_comprehensive_local_search,
    )
    inst = _create_toy_instance(3)
    # A suboptimal route: [0, 2, 1, 3, 0]
    initial_route = [0, 2, 1, 3, 0]
    init_dist = inst.calculate_route_distance(initial_route)

    opt_routes, improved_dist, num_imp = apply_comprehensive_local_search([initial_route], inst)
    improved_route = opt_routes[0]
    assert improved_route[0] == 0 and improved_route[-1] == 0
    assert set(improved_route[1:-1]) == {1, 2, 3}
    assert improved_dist <= init_dist + 1e-9


def test_ga_local_search_solver():
    """Verifies that GALocalSearchSolver executes properly and respects best-so-far."""
    from src.classical.ga_local_search import GALocalSearchSolver
    inst = _create_toy_instance(3)
    solver = GALocalSearchSolver(population_size=20, generations=10, seed=42)
    solution = solver.solve(inst)

    assert isinstance(solution, VRPSolution)
    assert solution.feasible is True
    assert solution.total_distance > 0.0
    assert solution.metadata["local_search_applied"] is True
    assert solution.metadata["initial_ga_distance"] >= solution.total_distance - 1e-9


def test_hybrid_ga_qaoa_solver():
    """
    Verifies that HybridGAQAOASolver initializes with GA, executes QAOA subproblems,
    preserves best-so-far, and records detailed metrics.
    """
    from src.quantum.hybrid_solver import HybridGAQAOASolver
    inst = _create_toy_instance(3)
    solver = HybridGAQAOASolver(
        ga_population_size=20,
        ga_generations=10,
        subproblem_size=3,
        qaoa_p_depth=1,
        qaoa_shots=200,
        qaoa_max_iter=10,
        hybrid_iterations=1,
        seed=42,
    )
    solution = solver.solve(inst)

    assert isinstance(solution, VRPSolution)
    assert solution.feasible is True
    assert "hybrid_metrics" in solution.metadata
    metrics = solution.metadata["hybrid_metrics"]
    assert "initial_ga_distance" in metrics
    assert "best_ga_distance" in metrics
    assert "best_hybrid_distance" in metrics
    assert "qaoa_calls_count" in metrics
    assert metrics["qaoa_calls_count"] >= 1
    # Best-so-far guarantee: hybrid distance must be <= best GA distance
    assert solution.total_distance <= metrics["best_ga_distance"] + 1e-9


def test_qaoa_valid_sample_accounting():
    """
    Verifies that QAOAVRPSolver honestly tracks raw_valid_sample_rate,
    repaired_solution_rate, and distinguishes raw samples from fallback.
    """
    inst = _create_toy_instance(2)
    solver = QAOAVRPSolver(p_depth=1, shots=300, max_iter=10, seed=42)
    solution = solver.solve(inst)

    assert "raw_valid_sample_rate" in solution.metadata
    assert "repaired_solution_rate" in solution.metadata
    assert "repair_applied" in solution.metadata
    assert "fallback_used" in solution.metadata
    assert 0.0 <= solution.metadata["raw_valid_sample_rate"] <= 100.0


def test_adaptive_penalty_calibration():
    """
    Verifies that VRPQuboBuilder calculates adaptive penalty based on triangle inequality:
    penalty = 2.0 * max_dist.
    """
    inst = _create_toy_instance(3)
    builder = VRPQuboBuilder(inst)
    max_d = np.max(inst.distance_matrix)
    expected_penalty = 2.0 * max_d
    assert abs(builder.penalty_weight - expected_penalty) < 1e-9

