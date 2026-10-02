"""
Dataset loader for VRP and EVRP benchmark instances.
Provides loading interfaces and deterministic toy instances for debugging and testing.
"""
from pathlib import Path
from typing import Any, Dict, Union, Optional
import numpy as np

from src.vrp.distance import compute_distance_matrix
from src.vrp.instance import VRPInstance


class DataLoader:
    """
    Handles loading raw benchmark files and generating small toy VRP instances.
    """

    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)

    def load_raw_instance(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads raw dataset instance file.

        Args:
            file_path (Union[str, Path]): Path to benchmark file.

        Returns:
            Dict[str, Any]: Parsed raw dataset contents.
        """
        raise NotImplementedError("TODO: Raw benchmark dataset parser to be implemented.")

    def create_toy_vrp_instance(self, num_nodes: int = 4, seed: int = 42) -> Dict[str, Any]:
        """
        Generates a small synthetic 2D Euclidean VRP instance.

        Args:
            num_nodes (int): Total number of nodes including depot (node 0).
            seed (int): Random seed for coordinate generation.

        Returns:
            Dict[str, Any]: Dictionary containing node coordinates, demands, and distance matrix.
        """
        np.random.seed(seed)
        coords = np.random.rand(num_nodes, 2) * 10.0  # Coordinates in 10x10 grid
        dist_matrix = compute_distance_matrix(coords, metric="euclidean")
        demands = np.zeros(num_nodes)
        if num_nodes > 1:
            demands[1:] = np.random.randint(1, 5, size=num_nodes - 1).astype(float)

        return {
            "num_nodes": num_nodes,
            "depot_idx": 0,
            "coordinates": coords,
            "distance_matrix": dist_matrix,
            "demands": demands,
            "vehicle_capacity": 15.0,
            "num_vehicles": 1,
        }

    def get_deterministic_toy_instance(self, num_customers: int = 3) -> Dict[str, Any]:
        """
        Provides small deterministic VRP toy instances (3, 4, or 5 customers)
        with fixed integer 2D coordinates and demands for easy verification.

        Args:
            num_customers (int): Number of customer nodes (3, 4, or 5).

        Returns:
            Dict[str, Any]: Instance data dictionary.
        """
        if num_customers == 3:
            # 4 total nodes (Depot + 3 Customers)
            coords = np.array([
                [0.0, 0.0],  # 0: Depot
                [0.0, 3.0],  # 1: C1 (Dist to depot = 3.0)
                [4.0, 0.0],  # 2: C2 (Dist to depot = 4.0)
                [4.0, 3.0],  # 3: C3 (Dist to depot = 5.0)
            ], dtype=float)
            demands = np.array([0.0, 2.0, 3.0, 4.0], dtype=float)
            vehicle_capacity = 10.0

        elif num_customers == 4:
            # 5 total nodes (Depot + 4 Customers)
            coords = np.array([
                [0.0, 0.0],  # 0: Depot
                [0.0, 3.0],  # 1: C1
                [4.0, 0.0],  # 2: C2
                [4.0, 3.0],  # 3: C3
                [0.0, 4.0],  # 4: C4
            ], dtype=float)
            demands = np.array([0.0, 2.0, 3.0, 4.0, 2.0], dtype=float)
            vehicle_capacity = 10.0

        elif num_customers == 5:
            # 6 total nodes (Depot + 5 Customers)
            coords = np.array([
                [0.0, 0.0],  # 0: Depot
                [0.0, 3.0],  # 1: C1
                [4.0, 0.0],  # 2: C2
                [4.0, 3.0],  # 3: C3
                [0.0, 4.0],  # 4: C4
                [3.0, 0.0],  # 5: C5
            ], dtype=float)
            demands = np.array([0.0, 2.0, 3.0, 4.0, 2.0, 1.0], dtype=float)
            vehicle_capacity = 12.0

        else:
            raise ValueError(f"Supported deterministic toy num_customers are [3, 4, 5], got {num_customers}.")

        dist_matrix = compute_distance_matrix(coords, metric="euclidean")
        num_nodes = len(coords)

        return {
            "num_nodes": num_nodes,
            "num_customers": num_customers,
            "depot_idx": 0,
            "coordinates": coords,
            "distance_matrix": dist_matrix,
            "demands": demands,
            "vehicle_capacity": vehicle_capacity,
            "num_vehicles": 1,
        }

    def load_vrp_instance(
        self,
        instance_type: str = "deterministic",
        num_customers: int = 3,
        num_nodes: int = 4,
        seed: int = 42
    ) -> VRPInstance:
        """
        Creates and returns a VRPInstance object.

        Args:
            instance_type (str): "deterministic" or "synthetic".
            num_customers (int): Customer count for deterministic instances (3, 4, or 5).
            num_nodes (int): Total node count for synthetic instances.
            seed (int): Seed for synthetic generation.

        Returns:
            VRPInstance: Instantiated VRP problem instance.
        """
        if instance_type == "deterministic":
            data = self.get_deterministic_toy_instance(num_customers=num_customers)
        elif instance_type == "synthetic":
            data = self.create_toy_vrp_instance(num_nodes=num_nodes, seed=seed)
        else:
            raise ValueError(f"Unknown instance_type '{instance_type}'. Choose 'deterministic' or 'synthetic'.")

        return VRPInstance(
            distance_matrix=data["distance_matrix"],
            coordinates=data["coordinates"],
            depot_idx=data["depot_idx"],
            demands=data["demands"],
            vehicle_capacity=data["vehicle_capacity"],
            num_vehicles=data["num_vehicles"],
        )
