"""
EVRP Problem Instance and Energy Modeling Representation.
Extends standard VRPInstance with battery state of charge (SoC), load-dependent energy consumption,
charging stations, charging duration, charging pricing, and leg-by-leg battery validation.
"""
from typing import List, Optional, Dict, Any, Union, Set
import numpy as np
from src.vrp.instance import VRPInstance


class EVRPInstance(VRPInstance):
    """
    Extends VRPInstance with comprehensive Electric Vehicle (EV) routing constraints.
    Supports payload capacity, battery SoC, charging stations, and load-dependent energy rate.
    """

    def __init__(
        self,
        distance_matrix: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
        battery_capacity: float = 100.0,
        energy_consumption_rate: float = 0.25,
        charging_stations: Optional[List[int]] = None,
        *,
        initial_battery: Optional[float] = None,
        battery_reserve: Optional[float] = None,
        load_factor: float = 0.2,
        charging_power: float = 50.0,
        charging_price_per_kwh: float = 0.25,
        vehicle_speed_kmh: float = 40.0,
        num_vehicles: int = 1,
        depot_idx: int = 0,
        demands: Optional[np.ndarray] = None,
        vehicle_capacity: Optional[float] = None,
        capacities: Optional[List[float]] = None,
        distance_metric: str = "euclidean",
        customer_subset: Optional[List[int]] = None,
        station_powers: Optional[Dict[int, float]] = None,
        station_prices: Optional[Dict[int, float]] = None,
        base_energy_rate: Optional[float] = None,
        charging_price: Optional[float] = None,
        speed: Optional[float] = None,
        **kwargs,
    ):
        """
        Args:
            distance_matrix (Optional[np.ndarray]): 2D distance matrix (N x N).
            coordinates (Optional[np.ndarray]): (N x 2) array of node coordinates.
            battery_capacity (float): Maximum battery energy capacity in kWh.
            energy_consumption_rate (float): Base energy consumption per unit distance (kWh/km).
            charging_stations (Optional[List[int]]): Node indices corresponding to charging stations.
            initial_battery (Optional[float]): Initial battery state of charge (kWh). Defaults to battery_capacity.
            battery_reserve (Optional[float]): Minimum allowable battery level (kWh). Defaults to 5% of capacity.
            load_factor (float): Multiplier parameter alpha for load-dependent consumption:
                                rate(L) = rate_base * (1 + alpha * (L / Q)).
            charging_power (float): Default charging power in kW.
            charging_price_per_kwh (float): Default electricity price ($/kWh).
            vehicle_speed_kmh (float): Average travel speed (km/h) for travel-time calculations.
            num_vehicles (int): Available homogeneous EV fleet size.
            depot_idx (int): Depot node identifier (default 0).
            demands (Optional[np.ndarray]): Node demands array.
            vehicle_capacity (Optional[float]): Maximum payload capacity per vehicle.
            capacities (Optional[List[float]]): Per-vehicle capacity list.
            distance_metric (str): Distance metric if coordinates supplied.
            customer_subset (Optional[List[int]]): Explicit customer node indices if subsetting.
            station_powers (Optional[Dict[int, float]]): Per-station charging power in kW.
            station_prices (Optional[Dict[int, float]]): Per-station electricity price in $/kWh.
        """
        super().__init__(
            distance_matrix=distance_matrix,
            coordinates=coordinates,
            num_vehicles=num_vehicles,
            depot_idx=depot_idx,
            demands=demands,
            vehicle_capacity=vehicle_capacity,
            capacities=capacities,
            distance_metric=distance_metric,
        )

        self.battery_capacity = float(battery_capacity)
        self.initial_battery = float(initial_battery) if initial_battery is not None else float(battery_capacity)
        self.battery_reserve = float(battery_reserve) if battery_reserve is not None else float(0.05 * battery_capacity)
        effective_rate = base_energy_rate if base_energy_rate is not None else energy_consumption_rate
        self.base_energy_rate = float(effective_rate)
        self.energy_consumption_rate = self.base_energy_rate
        self.load_factor = float(load_factor)
        self.default_charging_power = float(charging_power)
        effective_chg_price = charging_price if charging_price is not None else charging_price_per_kwh
        self.default_charging_price = float(effective_chg_price)
        effective_speed = speed if speed is not None else vehicle_speed_kmh
        self.vehicle_speed_kmh = float(effective_speed)

        # Categorize node types
        self.charging_stations: Set[int] = set(charging_stations) if charging_stations else set()
        self.station_powers: Dict[int, float] = station_powers or {}
        self.station_prices: Dict[int, float] = station_prices or {}

        self._customer_subset = list(customer_subset) if customer_subset is not None else None

    @property
    def customer_indices(self) -> List[int]:
        """Returns customer node indices (excluding depot and charging stations)."""
        if self._customer_subset is not None:
            return list(self._customer_subset)
        return [
            i for i in range(self.num_nodes)
            if i != self.depot_idx and i not in self.charging_stations
        ]

    @property
    def num_customers(self) -> int:
        """Returns number of customer nodes."""
        return len(self.customer_indices)

    def is_charging_station(self, node_idx: int) -> bool:
        """Returns True if node_idx is a charging station."""
        return node_idx in self.charging_stations

    def is_customer(self, node_idx: int) -> bool:
        """Returns True if node_idx is a delivery customer."""
        return node_idx in self.customer_indices

    def is_depot(self, node_idx: int) -> bool:
        """Returns True if node_idx is the central depot."""
        return node_idx == self.depot_idx

    def get_station_power(self, station_idx: int) -> float:
        """Returns charging power (kW) for given charging station."""
        return float(self.station_powers.get(station_idx, self.default_charging_power))

    def get_station_price(self, station_idx: int) -> float:
        """Returns electricity price ($/kWh) for given charging station."""
        return float(self.station_prices.get(station_idx, self.default_charging_price))

    def calculate_leg_energy(
        self, from_node: int, to_node: int, current_load: float = 0.0
    ) -> float:
        """
        Calculates energy consumed (kWh) traversing edge (from_node, to_node).
        Incorporates distance and vehicle payload load factor:
          Energy = Distance * Base_Rate * (1 + load_factor * (Load / Capacity))

        Args:
            from_node (int): Source node index.
            to_node (int): Destination node index.
            current_load (float): Current payload load carried by EV (units or kg).

        Returns:
            float: Energy consumed in kWh.
        """
        dist = float(self.distance_matrix[from_node, to_node])
        max_cap = self.capacities[0] if (self.capacities and self.capacities[0] > 0) else 100.0
        load_ratio = max(0.0, float(current_load)) / max(1e-6, max_cap)
        effective_rate = self.base_energy_rate * (1.0 + self.load_factor * load_ratio)
        return float(dist * effective_rate)

    def calculate_leg_time(self, from_node: int, to_node: int) -> float:
        """
        Calculates travel time in hours along edge (from_node, to_node) based on speed.
        """
        dist = float(self.distance_matrix[from_node, to_node])
        speed = max(1.0, self.vehicle_speed_kmh)
        return float(dist / speed)

    def evaluate_route_ev(self, route: List[int]) -> Dict[str, Any]:
        """
        Performs a full, rigorous leg-by-leg EV simulation of a vehicle route.
        Tracks payload capacity, battery SoC after every leg, recharges at stations,
        charging costs, charging times, and reports all constraint violations.

        Args:
            route (List[int]): Route starting and ending at depot_idx.

        Returns:
            Dict[str, Any]: Comprehensive evaluation report including:
                - feasible: bool
                - depot_valid: bool
                - capacity_valid: bool
                - battery_valid: bool
                - total_distance: float (km)
                - total_energy_consumed: float (kWh)
                - total_energy_charged: float (kWh)
                - total_charging_cost: float ($)
                - total_charging_time: float (hours)
                - total_travel_time: float (hours)
                - num_charging_stops: int
                - visited_customers: List[int]
                - leg_trace: List[Dict] with step-by-step telemetry
                - violations: List[str]
        """
        if not route or len(route) < 2:
            return {
                "feasible": False,
                "depot_valid": False,
                "capacity_valid": False,
                "battery_valid": False,
                "total_distance": 0.0,
                "total_energy_consumed": 0.0,
                "total_energy_charged": 0.0,
                "total_charging_cost": 0.0,
                "total_charging_time": 0.0,
                "total_travel_time": 0.0,
                "num_charging_stops": 0,
                "visited_customers": [],
                "leg_trace": [],
                "violations": ["Route must contain at least depot departure and arrival."],
            }

        depot_valid = (route[0] == self.depot_idx and route[-1] == self.depot_idx)
        violations = []
        if not depot_valid:
            violations.append(f"Route must start and end at depot {self.depot_idx}.")

        # Initial vehicle payload is the sum of demands for all customer nodes visited
        visited_customers = [n for n in route if self.is_customer(n)]
        total_customer_demand = sum(float(self.demands[c]) for c in visited_customers)
        max_cap = self.capacities[0] if (self.capacities and self.capacities[0] > 0) else float("inf")

        capacity_valid = total_customer_demand <= max_cap + 1e-6
        if not capacity_valid:
            violations.append(
                f"Payload demand ({total_customer_demand:.2f}) exceeds capacity ({max_cap:.2f})."
            )

        current_load = float(total_customer_demand)
        current_battery = float(self.initial_battery)

        total_distance = 0.0
        total_energy_consumed = 0.0
        total_energy_charged = 0.0
        total_charging_cost = 0.0
        total_charging_time = 0.0
        total_travel_time = 0.0
        num_charging_stops = 0
        battery_valid = True

        leg_trace = []

        for idx in range(len(route) - 1):
            u = route[idx]
            v = route[idx + 1]

            d = float(self.distance_matrix[u, v])
            t_travel = self.calculate_leg_time(u, v)
            e_consumed = self.calculate_leg_energy(u, v, current_load=current_load)

            b_before = current_battery
            b_after_travel = b_before - e_consumed

            total_distance += d
            total_travel_time += t_travel
            total_energy_consumed += e_consumed

            is_battery_violation = (b_after_travel < self.battery_reserve - 1e-6)
            if is_battery_violation:
                battery_valid = False
                violations.append(
                    f"Leg {idx} ({u} -> {v}): Battery dropped to {b_after_travel:.2f} kWh below reserve {self.battery_reserve:.2f} kWh."
                )

            # Customer service: delivery decreases onboard payload
            if self.is_customer(v):
                current_load = max(0.0, current_load - float(self.demands[v]))

            # Charging station visit or depot return: recharge vehicle
            e_charged = 0.0
            t_charge = 0.0
            c_charge = 0.0
            b_final = b_after_travel

            if self.is_charging_station(v) or self.is_depot(v):
                if b_after_travel < self.battery_capacity:
                    e_charged = float(self.battery_capacity - b_after_travel)
                    power = self.get_station_power(v)
                    price = self.get_station_price(v)
                    t_charge = e_charged / max(1.0, power)
                    c_charge = e_charged * price

                    total_energy_charged += e_charged
                    total_charging_time += t_charge
                    total_charging_cost += c_charge

                    if self.is_charging_station(v):
                        num_charging_stops += 1

                    b_final = float(self.battery_capacity)

            current_battery = b_final

            leg_trace.append({
                "step": idx,
                "from_node": int(u),
                "to_node": int(v),
                "distance": round(d, 4),
                "travel_time_hours": round(t_travel, 4),
                "energy_consumed_kwh": round(e_consumed, 4),
                "battery_before_kwh": round(b_before, 4),
                "battery_after_travel_kwh": round(b_after_travel, 4),
                "energy_charged_kwh": round(e_charged, 4),
                "battery_after_charge_kwh": round(b_final, 4),
                "charging_time_hours": round(t_charge, 4),
                "charging_cost_usd": round(c_charge, 4),
                "current_load": round(current_load, 2),
                "battery_violation": is_battery_violation,
            })

        overall_feasible = bool(depot_valid and capacity_valid and battery_valid)

        return {
            "feasible": overall_feasible,
            "depot_valid": depot_valid,
            "capacity_valid": capacity_valid,
            "battery_valid": battery_valid,
            "total_distance": round(total_distance, 4),
            "total_energy_consumed": round(total_energy_consumed, 4),
            "total_energy_charged": round(total_energy_charged, 4),
            "total_charging_cost": round(total_charging_cost, 4),
            "total_charging_time": round(total_charging_time, 4),
            "total_travel_time": round(total_travel_time, 4),
            "total_route_time": round(total_travel_time + total_charging_time, 4),
            "num_charging_stops": num_charging_stops,
            "visited_customers": visited_customers,
            "leg_trace": leg_trace,
            "violations": violations,
        }
