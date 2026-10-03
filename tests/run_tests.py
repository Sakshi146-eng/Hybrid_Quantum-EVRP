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
import tests.test_classical_base as tcb
import tests.test_genetic_algorithm as tga

if __name__ == "__main__":
    print("Running VRP Foundation, Classical Baseline, and Genetic Algorithm unit tests...")

    test_functions = [
        # Setup tests (6)
        ts.test_config_loader,
        ts.test_seed_reproducibility,
        ts.test_data_loader_toy_instance,
        ts.test_vrp_instance_cost,
        ts.test_evrp_feasibility_checker,
        ts.test_evaluation_metrics,
        # Classical VRP Foundation tests (10)
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
        # Phase 1 Classical Baseline Architecture tests (4)
        tcb.test_vrp_solution_structure_and_dict,
        tcb.test_abstract_base_solver_enforcement,
        tcb.test_random_baseline_solver_execution,
        tcb.test_solver_feasibility_check_infeasible_capacity,
        # Phase 2 Genetic Algorithm tests (8)
        tga.test_ga_solver_instantiation_and_name,
        tga.test_ga_solver_returns_vrp_solution,
        tga.test_ga_customer_coverage_and_no_duplicates,
        tga.test_ga_routes_start_and_end_at_depot,
        tga.test_ga_vehicle_capacity_enforcement,
        tga.test_ga_total_distance_validity,
        tga.test_ga_seed_reproducibility,
        tga.test_ga_on_toy_instances,
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
