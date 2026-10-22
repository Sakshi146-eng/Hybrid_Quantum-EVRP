"""
Demonstration of Genetic Algorithm (GA) Solver on CEC-2020 E-CVRP Benchmark Instances.
Loads a real benchmark instance, extracts classical CVRP data, executes the GA solver,
and records structured benchmark metrics.
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.classical.genetic_algorithm import GeneticAlgorithmSolver
from src.evaluation.benchmark import record_benchmark_result


def run_benchmark_demo(instance_path: str = "data/raw/e_cvrp/E-n29-k4-s7.evrp"):
    path = Path(instance_path)
    if not path.is_absolute():
        path = root_dir / path

    if not path.is_file():
        print(f"Benchmark file not found at: {path}")
        print("Please ensure dataset is downloaded in data/raw/e_cvrp/")
        return

    print("=" * 65)
    print("      GENETIC ALGORITHM (GA) CEC-2020 BENCHMARK DEMO      ")
    print("=" * 65)

    # 1. Load benchmark instance
    instance = load_e_cvrp_instance(path)
    inst_name = instance.raw_data.get("name", path.stem)
    tot_demand = float(instance.demands.sum())

    # 2. Print instance information
    print(f"Benchmark Instance: {inst_name}")
    print(f"Customers:          {instance.num_customers}")
    print(f"Vehicles:           {instance.num_vehicles}")
    print(f"Vehicle Capacity:   {instance.vehicle_capacity}")
    print(f"Total Demand:       {tot_demand}")
    print("-" * 65)

    # 3. Configure and run GeneticAlgorithmSolver
    solver = GeneticAlgorithmSolver(
        population_size=100,
        generations=200,
        crossover_rate=0.85,
        mutation_rate=0.25,
        tournament_size=4,
        elite_size=3,
        seed=42,
    )

    solution = solver.solve(instance)

    # Record result to data structure and file
    out_file = root_dir / "results" / "ga_benchmark_results.json"
    benchmark_res = record_benchmark_result(
        solution=solution,
        instance=instance,
        instance_name=inst_name,
        save_path=out_file
    )

    # 4. Print structured benchmark output
    print(f"Algorithm Name:    {solution.algorithm_name}")
    print(f"GA Total Distance: {solution.total_distance:.4f}")
    print(f"GA Runtime:        {solution.runtime_seconds:.6f} seconds")
    print(f"Feasible:          {'YES' if solution.feasible else 'NO'}")
    print("-" * 65)

    for i, route in enumerate(solution.routes, start=1):
        load = solution.vehicle_loads[i - 1] if i - 1 < len(solution.vehicle_loads) else 0.0
        route_str = " -> ".join(str(node) for node in route)
        cap_str = f"{load:.1f} / {instance.vehicle_capacity:.1f}" if instance.vehicle_capacity else f"{load:.1f}"
        print(f"Vehicle {i}:")
        print(f"  {route_str}")
        print(f"  Load: {cap_str}")
        print()

    print(f"Recorded benchmark result to: {out_file}")
    print("=" * 65)
    print("          BENCHMARK DEMO COMPLETED SUCCESSFULLY          ")
    print("=" * 65)


if __name__ == "__main__":
    benchmark_file = "data/raw/e_cvrp/E-n29-k4-s7.evrp"
    if len(sys.argv) > 1:
        benchmark_file = sys.argv[1]
    run_benchmark_demo(benchmark_file)
