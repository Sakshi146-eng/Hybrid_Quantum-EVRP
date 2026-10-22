"""
Electric Vehicle Routing Problem (EVRP) extension package.
Provides EV problem instance representations, leg-by-leg battery and energy simulation,
feasibility checkers, dynamic charging station insertion logic, and unified benchmarking.
"""
from src.evrp.instance import EVRPInstance
from src.evrp.feasibility import EVRPFeasibilityChecker
from src.evrp.charging import (
    find_best_charging_station,
    insert_charging_stations_if_needed,
    prune_redundant_charging_stops,
)
from src.evrp.cost import calculate_evrp_solution_cost

__all__ = [
    "EVRPInstance",
    "EVRPFeasibilityChecker",
    "find_best_charging_station",
    "insert_charging_stations_if_needed",
    "prune_redundant_charging_stops",
    "calculate_evrp_solution_cost",
]

