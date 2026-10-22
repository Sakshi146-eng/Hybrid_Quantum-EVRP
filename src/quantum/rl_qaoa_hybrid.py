"""
Integrated Reinforcement Learning (RL) and QAOA Hybrid EVRP Solvers.
Implements:
  1. RLQAOASolver (RL initialization + QAOA subproblem optimization + EV repair)
  2. FullHybridEVRPSolver (RL + QAOA + EV-aware Local Search + Best-so-Far Preservation)
Operates directly on EVRPInstance with leg-by-leg battery validation and charging station insertion.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.evrp.instance import EVRPInstance
from src.evrp.charging import insert_charging_stations_if_needed, prune_redundant_charging_stops
from src.evrp.cost import calculate_evrp_solution_cost
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution
from src.rl.agent import EVRLAgent
from src.classical.ev_local_search import EVLocalSearchSolver, apply_ev_local_search_to_route
from src.quantum.qubo import VRPQuboBuilder
from src.quantum.qaoa_solver import QAOAVRPSolver


class RLQAOASolver(BaseVRPSolver):
    """
    Hybrid RL + QAOA Solver (without classical local search).
    Uses RL agent policy to construct initial routes, identifies promising subproblem windows,
    and invokes parameterized QAOA to optimize customer sequencing.
    """

    def __init__(
        self,
        rl_agent: Optional[EVRLAgent] = None,
        subproblem_size: int = 3,
        qaoa_p_depth: int = 1,
        qaoa_shots: int = 1024,
        qaoa_max_iter: int = 25,
        hybrid_iterations: int = 2,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.rl_agent = rl_agent or EVRLAgent(seed=seed)
        self.subproblem_size = min(4, max(2, subproblem_size))
        self.qaoa_p_depth = qaoa_p_depth
        self.qaoa_shots = qaoa_shots
        self.qaoa_max_iter = qaoa_max_iter
        self.hybrid_iterations = hybrid_iterations
        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "RL + QAOA"

    def _optimize_subproblem_with_qaoa(
        self,
        instance: EVRPInstance,
        entry_node: int,
        cust_subset: List[int],
        exit_node: int,
    ) -> Tuple[Optional[List[int]], float, bool, Dict[str, Any]]:
        """
        Runs QAOA on a local customer subset between entry_node and exit_node.
        """
        builder = VRPQuboBuilder(
            instance=instance,
            entry_node=entry_node,
            exit_node=exit_node,
            customer_subset=cust_subset,
            penalty_multiplier=2.0,
        )

        sub_solver = QAOAVRPSolver(
            p_depth=self.qaoa_p_depth,
            shots=self.qaoa_shots,
            max_iter=self.qaoa_max_iter,
            seed=self.seed,
            penalty_multiplier=2.0,
        )

        _, qaoa_meta = sub_solver.solve_builder(builder)
        top_cands = qaoa_meta.get("top_candidates", [])

        D = instance.distance_matrix
        # Current baseline ordering cost
        current_cost = float(D[entry_node, cust_subset[0]])
        for idx in range(len(cust_subset) - 1):
            current_cost += float(D[cust_subset[idx], cust_subset[idx + 1]])
        current_cost += float(D[cust_subset[-1], exit_node])

        best_perm = None
        best_cost = current_cost
        is_improved = False
        raw_valid = False

        for cand in top_cands:
            bs = cand["bitstring"]
            raw_route, is_valid, _ = builder.decode_bitstring(bs)
            perm = None
            if is_valid and raw_route is not None:
                raw_valid = True
                perm = raw_route[1:-1]
            else:
                rep_route, _, _ = builder.repair_bitstring_to_route(bs)
                if rep_route is not None and len(rep_route) >= 2:
                    perm = rep_route[1:-1]

            if perm is not None and len(perm) == len(cust_subset) and set(perm) == set(cust_subset):
                local_cost = float(D[entry_node, perm[0]])
                for idx in range(len(perm) - 1):
                    local_cost += float(D[perm[idx], perm[idx + 1]])
                local_cost += float(D[perm[-1], exit_node])

                if local_cost < best_cost - 1e-4:
                    best_cost = local_cost
                    best_perm = perm
                    is_improved = True

        stats = {
            "raw_valid_found": raw_valid,
            "raw_valid_sample_rate": qaoa_meta.get("raw_valid_sample_rate", 0.0),
            "current_cost": current_cost,
            "best_cost": best_cost,
            "is_improved": is_improved,
            "qaoa_meta": qaoa_meta,
        }
        return best_perm, best_cost, is_improved, stats

    def _solve_instance(self, instance: EVRPInstance) -> List[List[int]]:
        t_start = time.perf_counter()

        # Step 1: RL route generation and EV charging repair
        t0_rl = time.perf_counter()
        raw_routes = self.rl_agent.generate_route_for_instance(instance, deterministic=True)
        initial_routes = []
        for r in raw_routes:
            rep, _ = insert_charging_stations_if_needed(r, instance)
            rep = prune_redundant_charging_stops(rep, instance)
            initial_routes.append(rep)
        t_rl = time.perf_counter() - t0_rl

        initial_cost_eval = calculate_evrp_solution_cost(instance, initial_routes)
        initial_cost = initial_cost_eval["total_cost"]

        best_routes = [list(r) for r in initial_routes]
        best_cost = float(initial_cost)
        best_eval = dict(initial_cost_eval)

        qaoa_calls = 0
        qaoa_improvements = 0
        qaoa_total_time = 0.0

        # Step 2: Iterative subproblem optimization via QAOA
        for iteration in range(self.hybrid_iterations):
            improved_in_iter = False

            # Part A: Intra-route sliding windows
            for r_idx in range(len(best_routes)):
                route = best_routes[r_idx]
                cust_indices_in_route = [i for i, n in enumerate(route) if instance.is_customer(n)]

                if len(cust_indices_in_route) < 2:
                    continue

                m = min(self.subproblem_size, len(cust_indices_in_route))

                for w_start in range(len(cust_indices_in_route) - m + 1):
                    window_pos = cust_indices_in_route[w_start : w_start + m]
                    cust_subset = [route[pos] for pos in window_pos]

                    entry_node = route[window_pos[0] - 1]
                    exit_node = route[window_pos[-1] + 1]

                    t_q0 = time.perf_counter()
                    opt_perm, _, is_imp, q_stats = self._optimize_subproblem_with_qaoa(
                        instance=instance,
                        entry_node=entry_node,
                        cust_subset=cust_subset,
                        exit_node=exit_node,
                    )
                    t_q_elapsed = time.perf_counter() - t_q0
                    qaoa_calls += 1
                    qaoa_total_time += t_q_elapsed

                    if is_imp and opt_perm is not None:
                        cand_route = (
                            route[: window_pos[0]] + opt_perm + route[window_pos[-1] + 1 :]
                        )
                        repaired_cand, is_batt_ok = insert_charging_stations_if_needed(cand_route, instance)

                        cand_routes = [list(r) for r in best_routes]
                        cand_routes[r_idx] = repaired_cand

                        cand_eval = calculate_evrp_solution_cost(instance, cand_routes)

                        if cand_eval["feasible"] and cand_eval["total_cost"] < best_cost - 1e-4:
                            best_routes = cand_routes
                            best_cost = cand_eval["total_cost"]
                            best_eval = cand_eval
                            qaoa_improvements += 1
                            improved_in_iter = True
                            route = repaired_cand
                            break

                if improved_in_iter:
                    break

            # Part B: Multi-vehicle consolidation via QAOA
            if not improved_in_iter and len(best_routes) > 1:
                all_custs = [n for r in best_routes for n in r if instance.is_customer(n)]
                total_dem = sum(float(instance.demands[c]) for c in all_custs)
                veh_capacity = instance.capacities[0] if instance.capacities else 100000.0

                if len(all_custs) <= min(4, self.subproblem_size + 1) and total_dem <= veh_capacity:
                    t_q0 = time.perf_counter()
                    opt_perm, _, is_imp, q_stats = self._optimize_subproblem_with_qaoa(
                        instance=instance,
                        entry_node=instance.depot_idx,
                        cust_subset=all_custs,
                        exit_node=instance.depot_idx,
                    )
                    t_q_elapsed = time.perf_counter() - t_q0
                    qaoa_calls += 1
                    qaoa_total_time += t_q_elapsed

                    perm_to_test = opt_perm if (is_imp and opt_perm is not None) else all_custs
                    cand_route = [instance.depot_idx] + perm_to_test + [instance.depot_idx]
                    repaired_cand, _ = insert_charging_stations_if_needed(cand_route, instance)
                    cand_eval = calculate_evrp_solution_cost(instance, [repaired_cand])

                    if cand_eval["feasible"] and cand_eval["total_cost"] < best_cost - 1e-4:
                        best_routes = [repaired_cand]
                        best_cost = cand_eval["total_cost"]
                        best_eval = cand_eval
                        qaoa_improvements += 1
                        improved_in_iter = True

            if not improved_in_iter:
                break

        from src.evrp.fleet_allocation import GlobalFleetCoordinator
        coordinator = GlobalFleetCoordinator(instance)
        best_routes = coordinator.enforce_fleet_limit(best_routes)
        best_eval = calculate_evrp_solution_cost(instance, best_routes)
        best_cost = best_eval["total_cost"]

        t_total = time.perf_counter() - t_start

        self.last_metadata = {
            "algorithm": self.algorithm_name,
            "initial_rl_cost": round(float(initial_cost), 4),
            "best_cost": round(float(best_cost), 4),
            "qaoa_calls": qaoa_calls,
            "qaoa_improvements": qaoa_improvements,
            "runtime_breakdown": {
                "rl_seconds": round(t_rl, 6),
                "qaoa_seconds": round(qaoa_total_time, 6),
                "total_seconds": round(t_total, 6),
            },
            "evrp_evaluation": best_eval,
            "feasible": best_eval["feasible"],
        }
        return best_routes

    def solve(self, instance: EVRPInstance) -> VRPSolution:
        t0 = time.perf_counter()
        routes = self._solve_instance(instance)
        runtime = time.perf_counter() - t0

        cost_eval = calculate_evrp_solution_cost(instance, routes)

        return VRPSolution(
            algorithm_name=self.algorithm_name,
            routes=routes,
            total_distance=cost_eval["total_distance"],
            feasible=cost_eval["feasible"],
            runtime_seconds=float(runtime),
            metadata={
                **self.last_metadata,
                "evrp_evaluation": cost_eval,
                "total_cost": cost_eval["total_cost"],
                "charging_cost": cost_eval["charging_cost"],
                "total_energy_kwh": cost_eval["total_energy_consumed_kwh"],
                "charging_stops": cost_eval["total_charging_stops"],
            },
        )


class FullHybridEVRPSolver(BaseVRPSolver):
    """
    Full Hybrid Solver combining Reinforcement Learning, Parameterized QAOA,
    EV-aware Local Search, and Strict Best-Feasible-Solution Management.
    """

    def __init__(
        self,
        rl_agent: Optional[EVRLAgent] = None,
        subproblem_size: int = 3,
        qaoa_p_depth: int = 1,
        qaoa_shots: int = 1024,
        qaoa_max_iter: int = 25,
        hybrid_iterations: int = 2,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.rl_qaoa_solver = RLQAOASolver(
            rl_agent=rl_agent,
            subproblem_size=subproblem_size,
            qaoa_p_depth=qaoa_p_depth,
            qaoa_shots=qaoa_shots,
            qaoa_max_iter=qaoa_max_iter,
            hybrid_iterations=hybrid_iterations,
            seed=seed,
            **kwargs,
        )
        self.ls_solver = EVLocalSearchSolver(seed=seed)
        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "Full Hybrid (RL + QAOA + EV-LS)"

    def _solve_instance(self, instance: EVRPInstance) -> List[List[int]]:
        t0 = time.perf_counter()

        # Step 1: Run RL + QAOA solver
        rl_qaoa_routes = self.rl_qaoa_solver._solve_instance(instance)
        rl_qaoa_meta = self.rl_qaoa_solver.last_metadata
        rl_qaoa_cost = rl_qaoa_meta.get("best_cost", float("inf"))

        # Step 2: Apply EV Local Search polishing (intra-route and inter-route)
        t1 = time.perf_counter()
        polished_routes, num_ls_imp = self.ls_solver.optimize_routes(rl_qaoa_routes, instance)
        t_ls = time.perf_counter() - t1

        cost_eval = calculate_evrp_solution_cost(instance, polished_routes)
        polished_cost = cost_eval["total_cost"]

        # Strict Best-so-Far Preservation
        if cost_eval["feasible"] and polished_cost <= rl_qaoa_cost:
            final_routes = polished_routes
            final_cost = polished_cost
            final_eval = cost_eval
        else:
            final_routes = rl_qaoa_routes
            final_cost = rl_qaoa_cost
            final_eval = rl_qaoa_meta.get("evrp_evaluation", cost_eval)

        from src.evrp.fleet_allocation import GlobalFleetCoordinator
        coordinator = GlobalFleetCoordinator(instance)
        final_routes = coordinator.enforce_fleet_limit(final_routes)
        final_eval = calculate_evrp_solution_cost(instance, final_routes)
        final_cost = final_eval["total_cost"]

        t_total = time.perf_counter() - t0

        self.last_metadata = {
            "algorithm": self.algorithm_name,
            "initial_rl_cost": rl_qaoa_meta.get("initial_rl_cost"),
            "rl_qaoa_cost": round(float(rl_qaoa_cost), 4),
            "final_cost": round(float(final_cost), 4),
            "qaoa_calls": rl_qaoa_meta.get("qaoa_calls", 0),
            "qaoa_improvements": rl_qaoa_meta.get("qaoa_improvements", 0),
            "local_search_improvements": num_ls_imp,
            "runtime_breakdown": {
                "rl_seconds": rl_qaoa_meta.get("runtime_breakdown", {}).get("rl_seconds", 0.0),
                "qaoa_seconds": rl_qaoa_meta.get("runtime_breakdown", {}).get("qaoa_seconds", 0.0),
                "local_search_seconds": round(t_ls, 6),
                "total_seconds": round(t_total, 6),
            },
            "evrp_evaluation": final_eval,
            "feasible": final_eval["feasible"],
        }
        return final_routes

    def solve(self, instance: EVRPInstance) -> VRPSolution:
        t0 = time.perf_counter()
        routes = self._solve_instance(instance)
        runtime = time.perf_counter() - t0

        cost_eval = calculate_evrp_solution_cost(instance, routes)

        return VRPSolution(
            algorithm_name=self.algorithm_name,
            routes=routes,
            total_distance=cost_eval["total_distance"],
            feasible=cost_eval["feasible"],
            runtime_seconds=float(runtime),
            metadata={
                **self.last_metadata,
                "evrp_evaluation": cost_eval,
                "total_cost": cost_eval["total_cost"],
                "charging_cost": cost_eval["charging_cost"],
                "total_energy_kwh": cost_eval["total_energy_consumed_kwh"],
                "charging_stops": cost_eval["total_charging_stops"],
            },
        )
