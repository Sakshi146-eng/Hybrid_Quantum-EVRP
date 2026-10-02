"""
EVRP problem instance representation.
Extends standard VRP with battery capacity, energy consumption rates, and charging station locations.
"""
from typing import List, Optional
import numpy as np
from src.vrp.instance import VRPInstance


class EVRPInstance(VRPInstance):
    """
    Extends VRPInstance with EV-specific parameters.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        battery_capacity: float,
        energy_consumption_rate: float,
        charging_stations: List[int],
        *,
        num_vehicles: int = 1,
        depot_idx: int = 0,
        demands: Optional[np.ndarray] = None,
        capacities: Optional[List[float]] = None,
    ):
        """
        Args:
            distance_matrix (np.ndarray): 2D distance matrix.
            battery_capacity (float): Maximum battery capacity (kWh or % state of charge).
            energy_consumption_rate (float): Energy consumed per unit distance.
            charging_stations (List[int]): Node indices corresponding to charging stations.
            num_vehicles (int): Available EVs.
            depot_idx (int): Depot node index.
            demands (Optional[np.ndarray]): Customer demand values.
            capacities (Optional[List[float]]): Vehicle load capacities.
        """
        super().__init__(
            distance_matrix=distance_matrix,
            num_vehicles=num_vehicles,
            depot_idx=depot_idx,
            demands=demands,
            capacities=capacities,
        )
        self.battery_capacity = battery_capacity
        self.energy_consumption_rate = energy_consumption_rate
        self.charging_stations = set(charging_stations)

    def is_charging_station(self, node_idx: int) -> bool:
        """
        Checks if a given node is a charging station.
        """
        return node_idx in self.charging_stations
