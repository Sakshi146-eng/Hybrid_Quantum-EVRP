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
import tests.test_e_cvrp_loader as tbl
import tests.test_backend_api as tapi
import tests.test_quantum_qubo_qaoa as tq
import tests.test_evrp_rl_qaoa as tevrp
import tests.test_hybrid_evrp_regression as treg
import tests.test_master_rl_qaoa_hybrid as tmast
import tests.test_scalability_and_fleet_allocation as tscale

if __name__ == "__main__":
    print("Running VRP Foundation, Classical Baseline, GA, Benchmark Dataset, FastAPI, Quantum QAOA, and Regression tests...")

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
        # CEC-2020 E-CVRP Benchmark Dataset Loader & GA Integration tests (8)
        tbl.test_benchmark_loader_valid_instance_loads,
        tbl.test_benchmark_depot_identification,
        tbl.test_benchmark_customer_count,
        tbl.test_benchmark_vehicle_count_and_capacity,
        tbl.test_benchmark_demands_and_coords_parsing,
        tbl.test_benchmark_instance_vrp_validation,
        tbl.test_benchmark_ga_execution_and_vrp_solution,
        tbl.test_benchmark_result_recording,
        # FastAPI Backend API tests (4)
        tapi.test_api_health,
        tapi.test_api_benchmark_instances_list,
        tapi.test_api_benchmark_instance_details,
        tapi.test_api_benchmark_ga_execution,
        # Quantum QUBO and QAOA tests (10)
        tq.test_qubo_variable_mapping,
        tq.test_qubo_matrix_construction,
        tq.test_qubo_energy_matches_actual_route_distance,
        tq.test_qubo_penalty_rejects_invalid_solutions,
        tq.test_route_decoding_and_validation,
        tq.test_ising_hamiltonian_eigenvalues,
        tq.test_exact_brute_force_solver,
        tq.test_qaoa_solver_execution_small_instance,
        tq.test_comparative_workflow_equivalent_inputs,
        tq.test_api_qaoa_and_compare_endpoints,
        tq.test_local_search_operators,
        tq.test_ga_local_search_solver,
        tq.test_hybrid_ga_qaoa_solver,
        tq.test_qaoa_valid_sample_accounting,
        tq.test_adaptive_penalty_calibration,
        # EVRP Model, RL, QAOA, and Hybrid Solvers (13)
        tevrp.test_ev_energy_consumption_and_load_factor,
        tevrp.test_ev_leg_by_leg_battery_tracking,
        tevrp.test_ev_charging_station_insertion,
        tevrp.test_ev_charging_station_pruning,
        tevrp.test_ev_feasibility_checker,
        tevrp.test_ev_cost_calculation,
        tevrp.test_ev_genetic_algorithm_solver,
        tevrp.test_ev_ga_local_search_solver,
        tevrp.test_ev_routing_gym_env,
        tevrp.test_ev_rl_agent_train_and_inference,
        tevrp.test_rl_qaoa_hybrid_solver,
        tevrp.test_full_hybrid_evrp_solver,
        tevrp.test_evrp_api_endpoints,
        # Hybrid EVRP Regression and Verification tests (7)
        treg.test_regression_rl_training_completion,
        treg.test_regression_model_checkpoint_save_and_reload,
        treg.test_regression_all_six_solvers_evaluated_correctly,
        treg.test_regression_qaoa_executes_on_subproblem,
        treg.test_regression_best_feasible_solution_preservation,
        treg.test_regression_improvement_percentages_math,
        treg.test_regression_backend_api_and_telemetry,
        # Master RL + QAOA Hybrid & Ablation tests (11)
        tmast.test_ev_qubo_formulation_and_marginals_decoding,
        tmast.test_rl_subproblem_agent_features_and_scoring,
        tmast.test_rl_qaoa_parameter_adaptation,
        tmast.test_qaoa_execution_and_exact_enumeration,
        tmast.test_candidate_diversity_and_multi_rollout,
        tmast.test_hierarchical_decomposition_large_instances,
        tmast.test_all_eight_solvers_in_benchmark,
        tmast.test_best_feasible_solution_preservation,
        tmast.test_timeout_and_fallback_handling,
        tmast.test_multi_seed_comparison_and_ablation_study,
        tmast.test_result_serialization_and_fastapi_endpoints,
        # Scalability & Fleet Allocation tests (9)
        tscale.test_customer_coverage_and_no_duplicates,
        tscale.test_vehicle_capacity_and_strict_fleet_limit,
        tscale.test_route_opening_with_dispatch_cost,
        tscale.test_large_instance_n20_feasibility,
        tscale.test_large_instance_n50_feasibility,
        tscale.test_rl_action_mask_validity,
        tscale.test_qubo_decoding_and_rejection,
        tscale.test_cost_consistency_with_recomputation,
        tscale.test_reproducibility_across_seeds,
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
