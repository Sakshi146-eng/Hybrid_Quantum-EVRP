"""
Unit tests for classical VRP foundation (Task 7).
Verifies data representation, coordinate storage, distance matrix calculations,
route distance calculations, depot start/end checks, invalid route handling, and capacity validation.
"""
import math
import numpy as np

from src.data.loader import DataLoader
from src.vrp.instance import VRPInstance
from src.vrp.distance import compute_distance_matrix, validate_distance_matrix
from src.vrp.validator import VRPValidator


def test_1_number_of_nodes():
    """Verify correct number of nodes stored in VRPInstance for 3, 4, 5 customer toy instances."""
    loader = DataLoader()
    for n_cust in [3, 4, 5]:
        inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=n_cust)
        assert inst.num_nodes == n_cust + 1
        assert inst.num_customers == n_cust
        assert len(inst.customer_indices) == n_cust


def test_2_coordinate_storage():
    """Verify correct coordinate storage in VRPInstance."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)
    assert inst.coordinates is not None
    assert inst.coordinates.shape == (4, 2)
    # Check exact depot coordinate (0, 0)
    np.testing.assert_array_equal(inst.coordinates[0], [0.0, 0.0])
    # Check C1 coordinate (0, 3)
    np.testing.assert_array_equal(inst.coordinates[1], [0.0, 3.0])


def test_3_distance_matrix_dimensions():
    """Verify distance matrix dimensions match (num_nodes, num_nodes)."""
    loader = DataLoader()
    for n_cust in [3, 4, 5]:
        inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=n_cust)
        expected_shape = (n_cust + 1, n_cust + 1)
        assert inst.distance_matrix.shape == expected_shape


def test_4_zero_diagonal():
    """Verify distance(i, i) == 0.0 for all nodes."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=5)
    diagonal = np.diag(inst.distance_matrix)
    np.testing.assert_array_almost_equal(diagonal, 0.0)


def test_5_symmetry_euclidean():
    """Verify distance(i, j) == distance(j, i) for symmetric Euclidean VRP."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=4)
    assert validate_distance_matrix(inst.distance_matrix) is True
    np.testing.assert_array_almost_equal(inst.distance_matrix, inst.distance_matrix.T)


def test_6_known_distance_values():
    """Verify known distance values for a simple 3-4-5 right triangle geometry."""
    # Depot (0,0), C1 (0,3), C2 (4,0), C3 (4,3)
    coords = np.array([
        [0.0, 0.0],
        [0.0, 3.0],
        [4.0, 0.0],
        [4.0, 3.0],
    ])
    dist_mat = compute_distance_matrix(coords, metric="euclidean")
    inst = VRPInstance(distance_matrix=dist_mat, coordinates=coords)

    # dist(0, 1) = 3.0 (along y-axis)
    assert math.isclose(inst.distance_matrix[0, 1], 3.0, rel_tol=1e-6)
    # dist(0, 2) = 4.0 (along x-axis)
    assert math.isclose(inst.distance_matrix[0, 2], 4.0, rel_tol=1e-6)
    # dist(0, 3) = sqrt(3^2 + 4^2) = 5.0 (hypotenuse)
    assert math.isclose(inst.distance_matrix[0, 3], 5.0, rel_tol=1e-6)
    # dist(1, 2) = sqrt(4^2 + 3^2) = 5.0
    assert math.isclose(inst.distance_matrix[1, 2], 5.0, rel_tol=1e-6)
    # dist(1, 3) = 4.0
    assert math.isclose(inst.distance_matrix[1, 3], 4.0, rel_tol=1e-6)
    # dist(2, 3) = 3.0
    assert math.isclose(inst.distance_matrix[2, 3], 3.0, rel_tol=1e-6)


def test_7_correct_route_distance_calculation():
    """Verify correct sum of distances along a candidate route: Depot -> C1 -> C3 -> Depot."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)
    # Route: 0 -> 1 -> 3 -> 0
    # dist(0,1) = 3.0, dist(1,3) = 4.0, dist(3,0) = 5.0
    # Total expected distance = 3.0 + 4.0 + 5.0 = 12.0
    route = [0, 1, 3, 0]
    total_dist = inst.calculate_route_distance(route)
    assert math.isclose(total_dist, 12.0, rel_tol=1e-6)


def test_8_depot_start_end_handling():
    """Verify validation of depot start and end points in candidate routes."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)

    # Valid route starting and ending at depot 0
    valid_route = [0, 1, 2, 0]
    val_valid = inst.validate_route(valid_route)
    assert val_valid["is_valid"] is True
    assert val_valid["depot_start_end"] is True

    # Invalid route not starting at depot
    no_depot_start = [1, 2, 0]
    val_no_start = inst.validate_route(no_depot_start)
    assert val_no_start["is_valid"] is False
    assert val_no_start["depot_start_end"] is False

    # Invalid route not ending at depot
    no_depot_end = [0, 1, 2]
    val_no_end = inst.validate_route(no_depot_end)
    assert val_no_end["is_valid"] is False
    assert val_no_end["depot_start_end"] is False


def test_9_invalid_route_node_handling():
    """Verify handling of invalid node indices (e.g. index out of bounds or negative)."""
    loader = DataLoader()
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)

    # Out of bounds positive node index (node 99 when num_nodes is 4)
    try:
        inst.calculate_route_distance([0, 99, 0])
        assert False, "Expected ValueError for node index out of bounds"
    except ValueError as e:
        assert "Invalid node index" in str(e)

    # Negative node index (-1)
    try:
        inst.calculate_route_distance([0, -1, 0])
        assert False, "Expected ValueError for negative node index"
    except ValueError as e:
        assert "Invalid node index" in str(e)

    # Check validator response for out of bounds node index
    val_out_of_bounds = inst.validate_route([0, 99, 0])
    assert val_out_of_bounds["is_valid"] is False
    assert val_out_of_bounds["valid_indices"] is False


def test_10_vehicle_capacity_validation():
    """Verify vehicle capacity enforcement on candidate route payload demands."""
    loader = DataLoader()
    # 3-customer instance: Demands: C1=2.0, C2=3.0, C3=4.0. Vehicle capacity = 10.0
    inst = loader.load_vrp_instance(instance_type="deterministic", num_customers=3)

    # Route visits C1 + C2 + C3: total demand = 2 + 3 + 4 = 9.0 <= 10.0 -> Valid
    route_feasible = [0, 1, 2, 3, 0]
    val_feasible = inst.validate_route(route_feasible)
    assert val_feasible["is_valid"] is True
    assert val_feasible["capacity_valid"] is True
    assert math.isclose(val_feasible["total_demand"], 9.0, rel_tol=1e-6)

    # Create instance with tight capacity = 5.0
    inst_tight = VRPInstance(
        distance_matrix=inst.distance_matrix,
        coordinates=inst.coordinates,
        demands=inst.demands,
        vehicle_capacity=5.0
    )
    # Route C1 + C2 + C3 has demand 9.0 > 5.0 capacity -> Invalid capacity
    val_infeasible = inst_tight.validate_route(route_feasible)
    assert val_infeasible["is_valid"] is False
    assert val_infeasible["capacity_valid"] is False
