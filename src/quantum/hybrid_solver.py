"""
Hybrid GA-QAOA Vehicle Routing Solver.
Combines Classical Genetic Algorithm global search with Quantum Approximate
Optimization Algorithm (QAOA) local subproblem optimization and Classical Local Search.

Staged Architecture:
  Stage A: Classical GA initialization
  Stage B: Promising local subproblem selection
  Stage C: Parameterized QAOA subproblem circuit simulation
  Stage D: Candidate reconstruction and constraint validation
  Stage E: Classical local improvement (2-Opt / Relocate)
  Stage F: Best-so-far candidate selection
  Stage G: Iterative refinement
"""
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.vrp.instance import VRPInstance
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.classical.local_search import (
    apply_comprehensive_local_search,
    calculate_route_cost,
)
from src.quantum.qubo import VRPQuboBuilder
from src.quantum.qaoa_solver import QAOAVRPSolver


class HybridGAQAOASolver(BaseVRPSolver):
    """
    Genuine Hybrid Quantum-Classical Solver integrating GA, QAOA subproblem
    optimization, and classical local search with strict best-so-far preservation.
    """

    def __init__(
        self,
        ga_population_size: int = 50,
        ga_generations: int = 100,
        ga_crossover_rate: float = 0.85,
        ga_mutation_rate: float = 0.25,
        subproblem_size: int = 4,
        max_hybrid_iterations: int = 3,
        qaoa_p_depth: int = 1,
        qaoa_shots: int = 1024,
        qaoa_max_iter: int = 35,
        use_local_search: bool = True,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        """
        Args:
            ga_population_size: GA chromosome population.
            ga_generations: GA evolutionary generations.
            subproblem_size: Number of customers in local QAOA subproblem (3 or 4).
            max_hybrid_iterations: Number of iterative refinement rounds.
            qaoa_p_depth: QAOA circuit depth layers p.
            qaoa_shots: Number of measurement shots per QAOA call.
            qaoa_max_iter: Classical COBYLA iterations per QAOA call.
            use_local_search: Whether to apply post-QAOA local search refinement.
            seed: Reproducibility random seed.
        """
        super().__init__(seed=seed, **kwargs)
        self.ga_population_size = ga_population_size
        self.ga_generations = ga_generations
        self.ga_crossover_rate = ga_crossover_rate
        self.ga_mutation_rate = ga_mutation_rate

        self.subproblem_size = min(4, max(2, int(subproblem_size)))
        self.max_hybrid_iterations = max(1, int(max_hybrid_iterations))
        self.qaoa_p_depth = max(1, int(qaoa_p_depth))
        self.qaoa_shots = max(100, int(qaoa_shots))
        self.qaoa_max_iter = max(10, int(qaoa_max_iter))
        self.use_local_search = use_local_search

        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "Hybrid GA-QAOA"

    def _select_promising_subproblems(
        self,
        route: List[int],
        dist_matrix: np.ndarray,
        window_size: int,
    ) -> List[Tuple[int, int, List[int], int, float]]:
        """
        Identifies and ranks candidate customer windows in the route for QAOA optimization.
        Returns tuples of (start_idx, end_idx, customer_subset, entry_node, exit_node, current_local_cost).
        """
        # route has format [depot, c_1, c_2, ..., c_k, depot]
        num_cust_in_route = len(route) - 2
        if num_cust_in_route < 2:
            return []

        actual_window = min(window_size, num_cust_in_route)
        candidates = []

        # Slide window across route positions: k from 1 to len(route) - 1 - actual_window
        for k in range(1, len(route) - actual_window):
            entry_node = route[k - 1]
            cust_subset = route[k : k + actual_window]
            exit_node = route[k + actual_window]

            # Current travel cost of this sub-path: entry -> c_1 -> ... -> c_m -> exit
            curr_cost = float(dist_matrix[entry_node, cust_subset[0]])
            for idx in range(len(cust_subset) - 1):
                curr_cost += float(dist_matrix[cust_subset[idx], cust_subset[idx + 1]])
            curr_cost += float(dist_matrix[cust_subset[-1], exit_node])

            candidates.append((k, k + actual_window, cust_subset, entry_node, exit_node, curr_cost))

        # Sort candidate windows by local cost in descending order (highest detour/cost first)
        candidates.sort(key=lambda item: item[5], reverse=True)
        return candidates

    def _optimize_subproblem_with_qaoa(
        self,
        instance: VRPInstance,
        cust_subset: List[int],
        entry_node: int,
        exit_node: int,
        current_cost: float,
    ) -> Tuple[Optional[List[int]], Optional[float], bool, Dict[str, Any]]:
        """
        Runs QAOA on the local subproblem to find an optimal re-ordering of cust_subset
        between entry_node and exit_node.

        Returns:
            Tuple: (best_permutation, best_local_cost, is_improved, qaoa_stats)
        """
        builder = VRPQuboBuilder(
            instance=instance,
            entry_node=entry_node,
            exit_node=exit_node,
            customer_subset=cust_subset,
            penalty_multiplier=2.0,
        )

        subproblem_solver = QAOAVRPSolver(
            p_depth=self.qaoa_p_depth,
            shots=self.qaoa_shots,
            max_iter=self.qaoa_max_iter,
            seed=self.seed,
            penalty_multiplier=2.0,
        )

        # Solve subproblem directly with QAOA on AerSimulator
        _, qaoa_meta = subproblem_solver.solve_builder(builder)

        # Extract top raw and repaired routes
        top_cands = qaoa_meta.get("top_candidates", [])
        best_cand_perm = None
        best_cand_cost = float("inf")
        is_improved = False
        raw_valid_found = False

        D = instance.distance_matrix

        # Evaluate candidate bitstrings specifically against this subproblem's endpoints
        for cand in top_cands:
            bs = cand["bitstring"]
            raw_route, is_valid, details = builder.decode_bitstring(bs)

            if is_valid and raw_route is not None:
                raw_valid_found = True
                # raw_route is [entry_node, p_1, ..., p_m, exit_node]
                perm = raw_route[1:-1]
                local_cost = float(D[entry_node, perm[0]])
                for idx in range(len(perm) - 1):
                    local_cost += float(D[perm[idx], perm[idx + 1]])
                local_cost += float(D[perm[-1], exit_node])

                if local_cost < best_cand_cost:
                    best_cand_cost = local_cost
                    best_cand_perm = perm

        # If no raw valid permutation was found, examine explicit repair
        repaired_cand_perm = None
        repaired_cand_cost = float("inf")
        if best_cand_perm is None and top_cands:
            top_bs = top_cands[0]["bitstring"]
            rep_route, was_rep, rep_details = builder.repair_bitstring_to_route(top_bs)
            perm = rep_route[1:-1]
            local_cost = float(D[entry_node, perm[0]])
            for idx in range(len(perm) - 1):
                local_cost += float(D[perm[idx], perm[idx + 1]])
            local_cost += float(D[perm[-1], exit_node])
            repaired_cand_cost = local_cost
            repaired_cand_perm = perm

        chosen_perm = best_cand_perm if best_cand_perm is not None else repaired_cand_perm
        chosen_cost = best_cand_cost if best_cand_perm is not None else repaired_cand_cost

        if chosen_perm is not None and chosen_cost < current_cost - 1e-4:
            is_improved = True

        stats = {
            "raw_valid_found": raw_valid_found,
            "raw_valid_sample_rate": qaoa_meta.get("raw_valid_sample_rate", 0.0),
            "best_cand_cost": best_cand_cost if best_cand_perm is not None else None,
            "repaired_cand_cost": repaired_cand_cost if repaired_cand_perm is not None else None,
            "current_cost": current_cost,
            "is_improved": is_improved,
            "qaoa_meta": qaoa_meta,
        }

        return chosen_perm, chosen_cost, is_improved, stats

    def _solve_instance(self, instance: VRPInstance) -> List[List[int]]:
        t_start_total = time.perf_counter()

        # -------------------------------------------------------------
        # Stage A: Classical GA Initialization
        # -------------------------------------------------------------
        t0_ga = time.perf_counter()
        ga_solver = GeneticAlgorithmSolver(
            population_size=self.ga_population_size,
            generations=self.ga_generations,
            crossover_rate=self.ga_crossover_rate,
            mutation_rate=self.ga_mutation_rate,
            seed=self.seed,
        )
        initial_ga_sol: VRPSolution = ga_solver.solve(instance)
        t_ga = time.perf_counter() - t0_ga

        current_routes = [list(r) for r in initial_ga_sol.routes]
        current_distance = float(initial_ga_sol.total_distance)

        # Strict Best-so-Far Preservation
        best_routes = [list(r) for r in current_routes]
        best_distance = float(current_distance)

        initial_ga_distance = float(current_distance)

        qaoa_calls = 0
        qaoa_improvements = 0
        hybrid_improvements = 0
        qaoa_total_time = 0.0
        local_search_total_time = 0.0
        subproblem_logs = []

        # If problem size <= subproblem_size, run QAOA on the entire instance directly
        if instance.num_customers <= self.subproblem_size and instance.num_customers >= 2:
            t0_q = time.perf_counter()
            full_qaoa = QAOAVRPSolver(
                p_depth=self.qaoa_p_depth,
                shots=self.qaoa_shots,
                max_iter=self.qaoa_max_iter,
                seed=self.seed,
                penalty_multiplier=2.0,
            )
            q_sol = full_qaoa.solve(instance)
            t_q = time.perf_counter() - t0_q
            qaoa_calls += 1
            qaoa_total_time += t_q

            if q_sol.feasible and q_sol.total_distance < best_distance - 1e-4:
                best_distance = float(q_sol.total_distance)
                best_routes = [list(r) for r in q_sol.routes]
                qaoa_improvements += 1
                hybrid_improvements += 1

            subproblem_logs.append({
                "iteration": 1,
                "type": "full_instance_qaoa",
                "qaoa_distance": float(q_sol.total_distance),
                "qaoa_feasible": q_sol.feasible,
                "qaoa_meta": q_sol.metadata,
            })
        else:
            # -------------------------------------------------------------
            # Stage B - G: Iterative Subproblem Optimization and Local Polish
            # -------------------------------------------------------------
            for iteration in range(1, self.max_hybrid_iterations + 1):
                iteration_improved = False

                for route_idx, route in enumerate(current_routes):
                    # Identify candidate windows in route
                    candidates = self._select_promising_subproblems(
                        route=route,
                        dist_matrix=instance.distance_matrix,
                        window_size=self.subproblem_size,
                    )

                    for start_k, end_k, cust_subset, entry_node, exit_node, curr_cost in candidates[:2]:
                        t0_q = time.perf_counter()
                        qaoa_calls += 1
                        opt_perm, opt_cost, is_improved, q_stats = self._optimize_subproblem_with_qaoa(
                            instance=instance,
                            cust_subset=cust_subset,
                            entry_node=entry_node,
                            exit_node=exit_node,
                            current_cost=curr_cost,
                        )
                        t_q = time.perf_counter() - t0_q
                        qaoa_total_time += t_q

                        if is_improved and opt_perm is not None:
                            qaoa_improvements += 1
                            # Stage D: Reconstruct candidate route
                            new_route = route[:start_k] + list(opt_perm) + route[end_k:]
                            val = instance.validate_route(new_route)

                            if val["is_valid"]:
                                candidate_routes = [list(r) for r in current_routes]
                                candidate_routes[route_idx] = new_route
                                cand_dist = instance.calculate_total_cost(candidate_routes)

                                if cand_dist < current_distance - 1e-4:
                                    current_routes = candidate_routes
                                    current_distance = cand_dist
                                    iteration_improved = True
                                    subproblem_logs.append({
                                        "iteration": iteration,
                                        "route_index": route_idx,
                                        "subproblem": cust_subset,
                                        "old_cost": curr_cost,
                                        "new_cost": opt_cost,
                                        "gain": curr_cost - opt_cost,
                                    })
                                    break
                    if iteration_improved:
                        break

                # Stage E: Classical Local Improvement (if enabled)
                if self.use_local_search:
                    t0_ls = time.perf_counter()
                    ls_routes, ls_dist, ls_imps = apply_comprehensive_local_search(
                        current_routes, instance
                    )
                    t_ls = time.perf_counter() - t0_ls
                    local_search_total_time += t_ls

                    if ls_dist < current_distance - 1e-4:
                        current_routes = ls_routes
                        current_distance = ls_dist
                        iteration_improved = True

                # Stage F: Best-so-Far Preservation
                if current_distance < best_distance - 1e-4:
                    best_distance = current_distance
                    best_routes = [list(r) for r in current_routes]
                    hybrid_improvements += 1

                if not iteration_improved:
                    # Early termination if no improvement in this round
                    break

        t_total = time.perf_counter() - t_start_total

        # Compute percentage improvement over initial GA baseline:
        # Improvement (%) = ((GA distance - Hybrid distance) / GA distance) * 100
        gain_percent = 0.0
        if initial_ga_distance > 1e-6:
            gain_percent = ((initial_ga_distance - best_distance) / initial_ga_distance) * 100.0

        self.last_metadata = {
            "initial_ga_distance": round(float(initial_ga_distance), 4),
            "best_ga_distance": round(float(initial_ga_distance), 4),
            "best_hybrid_distance": round(float(best_distance), 4),
            "improvement_over_ga_percent": round(float(gain_percent), 4),
            "qaoa_calls_count": qaoa_calls,
            "qaoa_improvements_count": qaoa_improvements,
            "hybrid_improvements_count": hybrid_improvements,
            "subproblem_size": self.subproblem_size,
            "max_hybrid_iterations": self.max_hybrid_iterations,
            "qaoa_depth_p": self.qaoa_p_depth,
            "qaoa_shots": self.qaoa_shots,
            "timing_breakdown": {
                "ga_initialization_seconds": round(t_ga, 6),
                "qaoa_total_seconds": round(qaoa_total_time, 6),
                "local_search_total_seconds": round(local_search_total_time, 6),
                "total_hybrid_seconds": round(t_total, 6),
            },
            "subproblem_logs": subproblem_logs,
            "device": "Qiskit Aer Simulator (Classical Emulation)",
        }

        return best_routes

    def solve(self, instance: VRPInstance) -> VRPSolution:
        solution = super().solve(instance)
        if hasattr(self, "last_metadata"):
            solution.metadata.update(self.last_metadata)
            solution.metadata["hybrid_metrics"] = dict(self.last_metadata)
        return solution
