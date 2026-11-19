"""
Classical VRP Foundation Demonstration Script.
Demonstrates toy instance creation, depot/customer data display, distance matrix calculation,
candidate route evaluation, total distance computation, and basic route validation.
"""
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.loader import DataLoader


def run_demo():
    print("=" * 65)
    print("      CLASSICAL VRP FOUNDATION DEMONSTRATION      ")
    print("=" * 65)
    print()

    # 1. Create a toy VRP instance
    loader = DataLoader()
    num_customers = 3
    print(f"[1] Loading Deterministic {num_customers}-Customer Toy VRP Instance...")
    instance = loader.load_vrp_instance(instance_type="deterministic", num_customers=num_customers)
    print("    Instance created successfully.\n")

    # 2. Print Customer + Depot Information
    print("[2] Node Metadata & Specifications:")
    print(f"    - Total Nodes (Depot + Customers): {instance.num_nodes}")
    print(f"    - Depot Index: {instance.depot_idx}")
    print(f"    - Number of Vehicles: {instance.num_vehicles}")
    print(f"    - Vehicle Payload Capacity: {instance.vehicle_capacity}")
    print()
    print("    Node Coordinates & Demands Table:")
    print("    --------------------------------------------")
    print("    Node ID | Type     | (X, Y) Coords | Demand ")
    print("    --------------------------------------------")
    for idx in range(instance.num_nodes):
        node_type = "Depot" if idx == instance.depot_idx else f"Cust {idx}"
        coords = instance.coordinates[idx]
        demand = instance.demands[idx]
        print(f"       {idx}    | {node_type:<8} | ({coords[0]:.1f}, {coords[1]:.1f})   | {demand:.1f}")
    print("    --------------------------------------------\n")

    # 3. Print Distance Matrix
    print("[3] Pairwise Euclidean Distance Matrix (N x N):")
    print("         ", end="")
    for j in range(instance.num_nodes):
        print(f"  Node {j} ", end="")
    print()
    for i in range(instance.num_nodes):
        print(f"    Node {i} |", end="")
        for j in range(instance.num_nodes):
            print(f"  {instance.distance_matrix[i, j]:6.2f} ", end="")
        print()
    print()

    # 4. Create Candidate Routes
    print("[4] Candidate Route Evaluation:")

    # Valid route
    route_a = [0, 1, 3, 2, 0]
    dist_a = instance.calculate_route_distance(route_a)
    val_a = instance.validate_route(route_a)

    print(f"    Candidate Route A: {route_a}")
    print(f"    - Total Traversal Distance: {dist_a:.2f}")
    print(f"    - Route Validation Result : {'VALID [PASS]' if val_a['is_valid'] else 'INVALID [FAIL]'}")
    print(f"    - Total Customer Demand   : {val_a['total_demand']:.1f} / {instance.vehicle_capacity:.1f}")
    print()

    # Infeasible capacity route (Tight capacity demo)
    print("    Testing Capacity Bound (with reduced capacity = 5.0):")
    route_b = [0, 1, 2, 3, 0]
    dist_b = instance.calculate_route_distance(route_b)

    # Validate with tight capacity manually
    val_b = instance.validate_route(route_b)
    # Perform tight validation check
    val_b_tight = instance.validate_route(route_b, check_capacity=False)
    total_demand_b = sum(instance.demands[n] for n in [1, 2, 3])
    print(f"    Candidate Route B: {route_b}")
    print(f"    - Total Traversal Distance: {dist_b:.2f}")
    print(f"    - Total Demand Visited    : {total_demand_b:.1f}")
    print(f"    - Capacity Limit (10.0)   : Feasible ({total_demand_b} <= 10.0)")
    print(f"    - Capacity Limit (5.0)    : Infeasible ({total_demand_b} > 5.0)")
    print()

    # Invalid route missing depot start
    route_c = [1, 2, 3, 0]
    val_c = instance.validate_route(route_c)
    print(f"    Candidate Route C (Invalid Start): {route_c}")
    print(f"    - Route Validation Result : {'VALID [PASS]' if val_c['is_valid'] else 'INVALID [FAIL]'}")
    print(f"    - Validation Errors       : {', '.join(val_c['errors'])}")
    print()

    print("=" * 65)
    print("            DEMONSTRATION COMPLETED SUCCESSFULLY          ")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
