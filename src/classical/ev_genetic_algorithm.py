"""
EV-Aware Genetic Algorithm (EV-GA) Solver.
Operates directly on EVRPInstance: uses chromosome permutations, capacity splitting,
charging station insertion, and evaluates fitness based on the unified EVRP objective.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.evrp.instance import EVRPInstance
from src.evrp.charging import insert_charging_stations_if_needed, prune_redundant_charging_stops
from src.evrp.cost import calculate_evrp_solution_cost
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution


class EVGeneticAlgorithmSolver(BaseVRPSolver):
    """
    EV-Aware Genetic Algorithm optimizing vehicle routing, payload capacity,
    battery consumption, and charging station visits.
    """

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 60,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        elite_size: int = 2,
        seed: Optional[int] = 42,
        w_dist: float = 1.0,
        w_charging: float = 1.0,
        w_vehicle: float = 50.0,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.population_size = max(10, population_size)
        self.generations = max(5, generations)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.w_dist = w_dist
        self.w_charging = w_charging
        self.w_vehicle = w_vehicle
        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "EV-Aware Classical GA"

    def _decode_chromosome_to_routes(
        self, chromosome: List[int], instance: EVRPInstance
    ) -> List[List[int]]:
        """Splits customer chromosome by payload capacity, then applies charging station insertion."""
        depot = instance.depot_idx
        capacity = instance.capacities[0] if instance.capacities else float("inf")
        demands = instance.demands

        raw_routes = []
        current_route = [depot]
        current_load = 0.0

        for customer in chromosome:
            d = float(demands[customer])
            if (current_load + d > capacity) and len(current_route) > 1:
                current_route.append(depot)
                raw_routes.append(current_route)
                current_route = [depot, customer]
                current_load = d
            else:
                current_route.append(customer)
                current_load += d

        if len(current_route) > 1:
            current_route.append(depot)
            raw_routes.append(current_route)
        else:
            raw_routes.append([depot, depot])

        # Step 2: Battery-aware charging station insertion
        ev_routes = []
        for r in raw_routes:
            repaired_r, _ = insert_charging_stations_if_needed(r, instance)
            ev_routes.append(repaired_r)

        return ev_routes

    def _evaluate_fitness(
        self, chromosome: List[int], instance: EVRPInstance
    ) -> Tuple[float, Dict[str, Any], List[List[int]]]:
        """Evaluates chromosome fitness based on the standardized EVRP objective."""
        routes = self._decode_chromosome_to_routes(chromosome, instance)
        cost_eval = calculate_evrp_solution_cost(
            instance,
            routes,
            w_dist=self.w_dist,
            w_charging=self.w_charging,
            w_vehicle=self.w_vehicle,
        )
        total_cost = cost_eval["total_cost"]
        fitness = 1.0 / (total_cost + 1e-6)
        return fitness, cost_eval, routes

    def _solve_instance(self, instance: EVRPInstance) -> List[List[int]]:
        t0 = time.perf_counter()
        if self.seed is not None:
            np.random.seed(self.seed)

        customers = list(instance.customer_indices)
        if not customers:
            return [[instance.depot_idx, instance.depot_idx]]

        # Initialize population
        population: List[List[int]] = []
        # Include sequential seed
        population.append(list(customers))
        for _ in range(self.population_size - 1):
            perm = list(customers)
            np.random.shuffle(perm)
            population.append(perm)

        best_routes = []
        best_cost = float("inf")
        best_eval = {}
        best_is_feasible = False

        for gen in range(self.generations):
            evaluated = [self._evaluate_fitness(chrom, instance) for chrom in population]
            fitnesses = [e[0] for e in evaluated]
            costs = [e[1]["total_cost"] for e in evaluated]

            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fit, gen_best_eval, gen_best_routes = evaluated[gen_best_idx]

            # Best-so-far preservation: prioritize feasible solutions
            is_feas = gen_best_eval["feasible"]
            cost_val = gen_best_eval["total_cost"]

            if is_feas:
                if not best_is_feasible or cost_val < best_cost:
                    best_cost = cost_val
                    best_routes = gen_best_routes
                    best_eval = gen_best_eval
                    best_is_feasible = True
            elif not best_is_feasible:
                if cost_val < best_cost:
                    best_cost = cost_val
                    best_routes = gen_best_routes
                    best_eval = gen_best_eval

            # Selection and Elitism
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = [population[i] for i in sorted_indices[: self.elite_size]]

            # Reproduction loop
            while len(new_population) < self.population_size:
                # Tournament selection
                t_idx1 = np.random.choice(self.population_size, size=self.tournament_size, replace=False)
                p1_idx = max(t_idx1, key=lambda i: fitnesses[i])
                parent1 = population[p1_idx]

                if np.random.rand() < self.crossover_rate:
                    t_idx2 = np.random.choice(self.population_size, size=self.tournament_size, replace=False)
                    p2_idx = max(t_idx2, key=lambda i: fitnesses[i])
                    parent2 = population[p2_idx]

                    # Order Crossover
                    n = len(parent1)
                    if n > 2:
                        cut1, cut2 = sorted(np.random.choice(n, size=2, replace=False))
                        child = [-1] * n
                        child[cut1 : cut2 + 1] = parent1[cut1 : cut2 + 1]
                        filled = set(child[cut1 : cut2 + 1])
                        p2_remaining = [c for c in parent2 if c not in filled]
                        idx = (cut2 + 1) % n
                        for c in p2_remaining:
                            while child[idx] != -1:
                                idx = (idx + 1) % n
                            child[idx] = c
                            idx = (idx + 1) % n
                    else:
                        child = list(parent1)
                else:
                    child = list(parent1)

                # Swap Mutation
                if np.random.rand() < self.mutation_rate and len(child) >= 2:
                    idx_a, idx_b = np.random.choice(len(child), size=2, replace=False)
                    child[idx_a], child[idx_b] = child[idx_b], child[idx_a]

                new_population.append(child)

            population = new_population

        t_elapsed = time.perf_counter() - t0

        self.last_metadata = {
            "algorithm": self.algorithm_name,
            "generations": self.generations,
            "population_size": self.population_size,
            "runtime_seconds": round(t_elapsed, 6),
            "evrp_evaluation": best_eval,
            "feasible": best_is_feasible,
        }

        return best_routes

    def solve(self, instance: EVRPInstance) -> VRPSolution:
        t0 = time.perf_counter()
        routes = self._solve_instance(instance)
        runtime = time.perf_counter() - t0

        cost_eval = calculate_evrp_solution_cost(
            instance, routes, w_dist=self.w_dist, w_charging=self.w_charging, w_vehicle=self.w_vehicle
        )

        sol = VRPSolution(
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
        return sol
