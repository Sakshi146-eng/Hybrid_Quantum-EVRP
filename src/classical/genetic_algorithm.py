"""
Genetic Algorithm (GA) Solver for Classical Vehicle Routing Problem (VRP).
Uses permutation-based chromosome representation, Order Crossover (OX), Swap Mutation,
Tournament Selection, Elitism, and Capacity-Conscious Route Decoding.
"""
from dataclasses import dataclass, field
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.vrp.instance import VRPInstance
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution


@dataclass
class GAConfig:
    """
    Configuration parameters for Genetic Algorithm VRP Solver.
    """
    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elite_size: int = 2
    seed: Optional[int] = 42


class GeneticAlgorithmSolver(BaseVRPSolver):
    """
    Genetic Algorithm implementation for standard Capacitated Vehicle Routing Problem (CVRP).
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        elite_size: int = 2,
        seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Args:
            population_size (int): Size of the chromosome population.
            generations (int): Number of evolutionary iterations.
            crossover_rate (float): Probability of applying crossover to selected parents.
            mutation_rate (float): Probability of mutating offspring.
            tournament_size (int): Number of candidates in tournament selection.
            elite_size (int): Number of top individuals preserved directly across generations.
            seed (Optional[int]): Random seed for reproducible runs.
            **kwargs: Extra hyperparameters.
        """
        super().__init__(seed=seed, **kwargs)
        self.config = GAConfig(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            elite_size=elite_size,
            seed=seed
        )

    @property
    def algorithm_name(self) -> str:
        return "Genetic Algorithm"

    def decode_chromosome_to_routes(
        self, chromosome: List[int], instance: VRPInstance
    ) -> List[List[int]]:
        """
        Greedy Split Strategy: Converts a customer permutation chromosome into valid vehicle routes
        while respecting vehicle capacity bounds.

        Strategy:
            1. Iterate through customer nodes in the chromosome order.
            2. Accumulate demand in the current vehicle route.
            3. If adding the next customer exceeds vehicle_capacity (or capacity constraint),
               close the current route with depot, start a new vehicle route from depot.
            4. Close final vehicle route with depot.

        Args:
            chromosome (List[int]): Permutation of customer node indices.
            instance (VRPInstance): Classical VRP problem instance.

        Returns:
            List[List[int]]: List of vehicle routes (each route starts & ends at depot_idx).
        """
        depot = instance.depot_idx
        capacity = instance.vehicle_capacity
        demands = instance.demands

        routes = []
        current_route = [depot]
        current_load = 0.0

        for customer in chromosome:
            cust_demand = float(demands[customer])

            # If single customer demand exceeds capacity itself (edge case), place in own route
            if capacity is not None and (current_load + cust_demand > capacity) and len(current_route) > 1:
                # Close current route
                current_route.append(depot)
                routes.append(current_route)
                # Start new route
                current_route = [depot, customer]
                current_load = cust_demand
            else:
                current_route.append(customer)
                current_load += cust_demand

        # Close final route
        if len(current_route) > 1:
            current_route.append(depot)
            routes.append(current_route)
        else:
            # Empty route fallback
            routes.append([depot, depot])

        return routes

    def _evaluate_fitness(
        self, chromosome: List[int], instance: VRPInstance
    ) -> Tuple[float, float, List[List[int]], bool]:
        """
        Evaluates the fitness of a customer permutation chromosome.

        Fitness = 1.0 / (total_distance + penalty + 1e-6)

        Args:
            chromosome (List[int]): Customer permutation.
            instance (VRPInstance): VRP problem instance.

        Returns:
            Tuple[float, float, List[List[int]], bool]: (fitness, total_distance, routes, is_feasible)
        """
        routes = self.decode_chromosome_to_routes(chromosome, instance)
        total_dist = instance.calculate_total_cost(routes)

        # Penalty calculation
        penalty = 0.0
        is_feasible = True

        # Check vehicle capacity constraints
        if instance.vehicle_capacity is not None:
            for r in routes:
                val = instance.validate_route(r)
                if not val["is_valid"]:
                    is_feasible = False
                    penalty += 500.0  # Large distance penalty for capacity overflow

        # Check max vehicles constraint if specified
        if len(routes) > instance.num_vehicles:
            is_feasible = False
            penalty += (len(routes) - instance.num_vehicles) * 1000.0

        score = total_dist + penalty
        fitness = 1.0 / (score + 1e-6)
        return fitness, total_dist, routes, is_feasible

    def _tournament_selection(
        self, population: List[List[int]], fitnesses: List[float]
    ) -> List[int]:
        """Performs tournament selection to select a parent chromosome."""
        pop_size = len(population)
        tourn_indices = np.random.choice(pop_size, size=self.config.tournament_size, replace=False)
        best_idx = max(tourn_indices, key=lambda i: fitnesses[i])
        return list(population[best_idx])

    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Order Crossover (OX1): Preserves order and relative positions of elements
        without creating invalid duplicate or missing customer permutations.
        """
        size = len(parent1)
        if size <= 2:
            return list(parent1)

        # Choose two distinct cut points
        cut1, cut2 = sorted(np.random.choice(size, size=2, replace=False))

        offspring = [None] * size
        # Copy slice from parent1
        offspring[cut1:cut2+1] = parent1[cut1:cut2+1]

        # Fill remaining positions using parent2 sequence
        copied_set = set(offspring[cut1:cut2+1])
        fill_vals = [item for item in parent2 if item not in copied_set]

        fill_idx = 0
        for i in range(size):
            if offspring[i] is None:
                offspring[i] = fill_vals[fill_idx]
                fill_idx += 1

        return offspring

    def _swap_mutation(self, chromosome: List[int]) -> List[int]:
        """Applies swap mutation on two distinct gene positions in chromosome."""
        mutated = list(chromosome)
        if len(mutated) >= 2:
            i, j = np.random.choice(len(mutated), size=2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def _solve_instance(self, instance: VRPInstance) -> List[List[int]]:
        """
        Core Genetic Algorithm evolution loop:
        1. Initialize population of random customer permutations.
        2. Evaluate fitness of each chromosome.
        3. Iterate through generations (Selection, OX Crossover, Swap Mutation, Elitism).
        4. Return best decoded vehicle routes.
        """
        customers = instance.customer_indices
        num_customers = len(customers)

        if num_customers == 0:
            return [[instance.depot_idx, instance.depot_idx]]

        pop_size = self.config.population_size
        generations = self.config.generations

        # 1. Initial Population Generation
        population = []
        for _ in range(pop_size):
            perm = list(customers)
            np.random.shuffle(perm)
            population.append(perm)

        best_chromosome = None
        best_fitness = -1.0
        best_routes = None
        best_distance = float("inf")

        history_distance = []

        # 2. Evolution Loop
        for gen in range(generations):
            # Evaluate fitness of population
            eval_results = [self._evaluate_fitness(chrom, instance) for chrom in population]
            fitnesses = [res[0] for res in eval_results]

            # Track best solution in generation
            for i, (fit, dist, rts, feasible) in enumerate(eval_results):
                if fit > best_fitness:
                    best_fitness = fit
                    best_chromosome = list(population[i])
                    best_routes = rts
                    best_distance = dist

            history_distance.append(float(best_distance))

            # 3. Elitism: preserve top elite_size individuals
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = [list(population[idx]) for idx in sorted_indices[:self.config.elite_size]]

            # 4. Generate remaining offspring
            while len(new_population) < pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)

                # Crossover
                if np.random.rand() < self.config.crossover_rate:
                    offspring = self._order_crossover(p1, p2)
                else:
                    offspring = list(p1)

                # Mutation
                if np.random.rand() < self.config.mutation_rate:
                    offspring = self._swap_mutation(offspring)

                new_population.append(offspring)

            population = new_population

        # Store convergence curve & metadata in self._last_metadata for solver wrapper
        self.last_metadata = {
            "best_distance_history": history_distance,
            "generations": generations,
            "population_size": pop_size,
            "crossover_rate": self.config.crossover_rate,
            "mutation_rate": self.config.mutation_rate,
            "elite_size": self.config.elite_size,
        }

        return best_routes if best_routes is not None else self.decode_chromosome_to_routes(customers, instance)

    def solve(self, instance: VRPInstance) -> VRPSolution:
        """Overridden solve wrapper to attach algorithm-specific GA metadata."""
        solution = super().solve(instance)
        if hasattr(self, "last_metadata"):
            solution.metadata.update(self.last_metadata)
        return solution
