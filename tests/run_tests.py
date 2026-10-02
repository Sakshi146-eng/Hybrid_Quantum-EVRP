"""
Lightweight test runner that executes test functions in tests/ without external dependencies.
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import tests.test_setup as ts
import tests.test_vrp as tv

if __name__ == "__main__":
    print("Running VRP Foundation and Setup verification unit tests...")
    
    test_functions = [
        # Setup tests
        ts.test_config_loader,
        ts.test_seed_reproducibility,
        ts.test_data_loader_toy_instance,
        ts.test_vrp_instance_cost,
        ts.test_evrp_feasibility_checker,
        ts.test_evaluation_metrics,
        # Task 7 VRP Foundation tests
        tv.test_1_number_of_nodes,
        tv.test_2_coordinate_storage,
        tv.test_3_distance_matrix_dimensions,
        tv.test_4_zero_diagonal,
        tv.test_5_symmetry_euclidean,
        tv.test_6_known_distance_values,
        tv.test_7_correct_route_distance_calculation,
        tv.test_8_depot_start_end_handling,
        tv.test_9_invalid_route_node_handling,
        tv.test_10_vehicle_capacity_validation,
    ]

    passed = 0
    failed = 0

    for test_fn in test_functions:
        fn_name = test_fn.__name__
        try:
            test_fn()
            print(f"  [PASS] {fn_name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {fn_name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nTest Summary: {passed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
