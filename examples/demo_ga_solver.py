"""
Demonstration of Genetic Algorithm (GA) Solver on Classical VRP Toy Instances.
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.loader import DataLoader
from src.classical.genetic_algorithm import GeneticAlgorithmSolver


def main():
    print("=" * 65)
    print("      GENETIC ALGORITHM (GA) CLASSICAL VRP DEMO      ")
    print("=" * 65)
    print()

    loader = DataLoader()
    solver = GeneticAlgorithmSolver(
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.2,
        seed=42
    )

    for n_cust in [3, 4, 5]:
        print(f"--- Solving {n_cust}-Customer VRP Toy Instance ---")
        instance = loader.load_vrp_instance(instance_type="deterministic", num_customers=n_cust)
        solution = solver.solve(instance)
        print(solution.summary())
        print()

    print("=" * 65)
    print("          GA DEMONSTRATION COMPLETED SUCCESSFULLY        ")
    print("=" * 65)


if __name__ == "__main__":
    main()
