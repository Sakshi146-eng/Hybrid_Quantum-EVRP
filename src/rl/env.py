"""
Gymnasium-compatible EVRP Routing Environment.
Provides comprehensive state representation, masked action space, and transparent reward modeling.
"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.evrp.instance import EVRPInstance
from src.evrp.charging import find_best_charging_station


class EVRoutingEnv(gym.Env):
    """
    Gymnasium environment for Electric Vehicle Routing Problem (EVRP).
    The agent acts as an EV driver, sequentially selecting the next destination node
    (unvisited customer, charging station, or depot) while managing payload and battery SoC.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        instance: EVRPInstance,
        max_nodes: int = 16,
        w_dist: float = 1.0,
        w_energy: float = 0.5,
        w_charging: float = 0.5,
        reward_customer_served: float = 5.0,
        reward_complete_tour: float = 15.0,
        penalty_infeasible: float = 20.0,
    ):
        """
        Args:
            instance (EVRPInstance): Active EVRP instance.
            max_nodes (int): Maximum nodes in fixed observation matrix (zero-padded).
            w_dist (float): Weight for distance penalty.
            w_energy (float): Weight for energy consumption penalty.
            w_charging (float): Weight for charging cost penalty.
            reward_customer_served (float): Bonus for successfully delivering to an unserved customer.
            reward_complete_tour (float): Bonus for completing all customer deliveries and returning to depot.
            penalty_infeasible (float): Penalty for battery depletion or capacity violation.
        """
        super().__init__()
        self.instance = instance
        self.max_nodes = max_nodes
        self.w_dist = w_dist
        self.w_energy = w_energy
        self.w_charging = w_charging
        self.reward_customer_served = reward_customer_served
        self.reward_complete_tour = reward_complete_tour
        self.penalty_infeasible = penalty_infeasible

        # Action Space: Discrete node selection [0 .. max_nodes - 1]
        self.action_space = spaces.Discrete(self.max_nodes)

        # Observation Space:
        # Global state (6 features) + Node features (max_nodes x 8 features)
        # Total observation dimension: 6 + max_nodes * 8
        self.obs_dim = 6 + self.max_nodes * 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.current_node: int = 0
        self.current_battery: float = 0.0
        self.current_load: float = 0.0
        self.visited_customers: set = set()
        self.visited_stations_count: int = 0
        self.route_history: List[int] = []
        self.current_step: int = 0
        self.max_steps: int = self.max_nodes * 3
        self.total_distance: float = 0.0
        self.total_energy: float = 0.0
        self.total_charging_cost: float = 0.0

    def set_instance(self, instance: EVRPInstance) -> None:
        """Updates the active EVRP instance."""
        self.instance = instance

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_node = self.instance.depot_idx
        self.current_battery = float(self.instance.initial_battery)

        # Initial payload is total demand of all customers to serve
        self.current_load = float(
            sum(self.instance.demands[c] for c in self.instance.customer_indices)
        )
        self.visited_customers = set()
        self.visited_stations_count = 0
        self.route_history = [self.current_node]
        self.current_step = 0
        self.total_distance = 0.0
        self.total_energy = 0.0
        self.total_charging_cost = 0.0

        obs = self._get_observation()
        info = {
            "action_mask": self.get_action_mask(),
            "route_history": list(self.route_history),
        }
        return obs, info

    def get_action_mask(self) -> np.ndarray:
        """
        Returns boolean action mask of shape (max_nodes,).
        True indicates action is feasible and valid; False indicates invalid action.
        """
        mask = np.zeros(self.max_nodes, dtype=bool)
        all_custs_served = len(self.visited_customers) == len(self.instance.customer_indices)

        for a in range(min(self.instance.num_nodes, self.max_nodes)):
            if a == self.instance.depot_idx:
                # Can return to depot if some customers served or battery allows
                e_depot = self.instance.calculate_leg_energy(
                    self.current_node, a, current_load=self.current_load
                )
                if self.current_battery - e_depot >= self.instance.battery_reserve - 1e-6:
                    mask[a] = True
                elif self.current_node == self.instance.depot_idx:
                    # At depot and no customers feasible: allow terminating
                    mask[a] = True

            elif self.instance.is_customer(a):
                if a in self.visited_customers:
                    continue  # Already visited
                # Check battery feasibility to customer a
                e_cust = self.instance.calculate_leg_energy(
                    self.current_node, a, current_load=self.current_load
                )
                if self.current_battery - e_cust < self.instance.battery_reserve - 1e-6:
                    continue  # Infeasible battery
                # Check that vehicle has battery left or a reachable station from a
                best_s = find_best_charging_station(
                    a, self.instance.depot_idx, self.current_battery - e_cust, self.current_load, self.instance
                )
                e_back = self.instance.calculate_leg_energy(a, self.instance.depot_idx, current_load=self.current_load)
                if (self.current_battery - e_cust - e_back >= self.instance.battery_reserve - 1e-6) or best_s is not None or not self.instance.charging_stations:
                    mask[a] = True

            elif self.instance.is_charging_station(a):
                # Station is valid if reachable and battery is not already >= 90% full
                if self.current_battery >= 0.90 * self.instance.battery_capacity:
                    continue
                e_station = self.instance.calculate_leg_energy(
                    self.current_node, a, current_load=self.current_load
                )
                if self.current_battery - e_station >= self.instance.battery_reserve - 1e-6:
                    mask[a] = True

        # Safety fallback: if no action is masked True, enable depot return
        if not np.any(mask):
            mask[self.instance.depot_idx] = True

        return mask

    def _get_observation(self) -> np.ndarray:
        """Constructs normalized state observation vector."""
        max_d = float(np.max(self.instance.distance_matrix)) if np.max(self.instance.distance_matrix) > 0 else 1.0
        max_b = max(1.0, float(self.instance.battery_capacity))
        max_cap = self.instance.capacities[0] if self.instance.capacities else 100.0

        coords = self.instance.coordinates
        curr_x, curr_y = (0.0, 0.0)
        if coords is not None and self.current_node < len(coords):
            curr_x, curr_y = coords[self.current_node]

        # Global features (6)
        global_features = [
            float(self.current_node) / max(1.0, float(self.instance.num_nodes)),
            curr_x / 100.0,
            curr_y / 100.0,
            self.current_battery / max_b,
            self.current_load / max(1.0, max_cap),
            len(self.visited_customers) / max(1.0, float(len(self.instance.customer_indices))),
        ]

        # Node features (max_nodes x 8)
        node_features = []
        for i in range(self.max_nodes):
            if i < self.instance.num_nodes:
                nx, ny = coords[i] if coords is not None else (0.0, 0.0)
                dist_curr = float(self.instance.distance_matrix[self.current_node, i]) / max_d
                dem = float(self.instance.demands[i]) / max(1.0, max_cap)
                is_station = 1.0 if self.instance.is_charging_station(i) else 0.0
                is_cust = 1.0 if self.instance.is_customer(i) else 0.0
                is_visited = 1.0 if i in self.visited_customers else 0.0
                is_active = 1.0
                node_features.extend([
                    is_active,
                    (nx - curr_x) / 100.0,
                    (ny - curr_y) / 100.0,
                    dist_curr,
                    dem,
                    is_cust,
                    is_station,
                    is_visited,
                ])
            else:
                # Padding node
                node_features.extend([0.0] * 8)

        full_obs = np.array(global_features + node_features, dtype=np.float32)
        return full_obs

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        mask = self.get_action_mask()

        # Reject invalid actions with high penalty
        if not mask[action] or action >= self.instance.num_nodes:
            # Force depot fallback and terminate
            reward = -self.penalty_infeasible
            terminated = True
            truncated = False
            info = {
                "feasible": False,
                "reason": f"Invalid action {action} chosen.",
                "route_history": list(self.route_history),
                "action_mask": mask,
            }
            return self._get_observation(), reward, terminated, truncated, info

        u = self.current_node
        v = int(action)

        dist = float(self.instance.distance_matrix[u, v])
        energy_leg = self.instance.calculate_leg_energy(u, v, current_load=self.current_load)

        self.total_distance += dist
        self.total_energy += energy_leg

        battery_after = self.current_battery - energy_leg
        reward = -(self.w_dist * dist * 0.05) - (self.w_energy * energy_leg * 0.05)

        # Battery violation check
        if battery_after < self.instance.battery_reserve - 1e-6:
            reward -= self.penalty_infeasible
            terminated = True
            truncated = False
            self.route_history.append(v)
            info = {
                "feasible": False,
                "reason": f"Battery violation on leg {u} -> {v}.",
                "route_history": list(self.route_history),
            }
            return self._get_observation(), reward, terminated, truncated, info

        # Customer visit
        if self.instance.is_customer(v):
            if v not in self.visited_customers:
                self.visited_customers.add(v)
                self.current_load = max(0.0, self.current_load - float(self.instance.demands[v]))
                reward += self.reward_customer_served

        # Charging station or depot recharge
        if self.instance.is_charging_station(v) or self.instance.is_depot(v):
            if battery_after < self.instance.battery_capacity:
                e_charged = self.instance.battery_capacity - battery_after
                price = self.instance.get_station_price(v)
                c_charge = e_charged * price
                self.total_charging_cost += c_charge
                reward -= (self.w_charging * c_charge * 0.1)
                battery_after = float(self.instance.battery_capacity)
            if self.instance.is_charging_station(v):
                self.visited_stations_count += 1

        self.current_node = v
        self.current_battery = battery_after
        self.route_history.append(v)

        # Termination conditions
        all_customers_served = (len(self.visited_customers) == len(self.instance.customer_indices))
        at_depot = (self.current_node == self.instance.depot_idx)

        terminated = False
        truncated = False

        if all_customers_served and at_depot and len(self.route_history) > 1:
            terminated = True
            reward += self.reward_complete_tour

        if self.current_step >= self.max_steps:
            truncated = True
            if not all_customers_served:
                unserved = len(self.instance.customer_indices) - len(self.visited_customers)
                reward -= (unserved * 5.0)

        info = {
            "feasible": True,
            "route_history": list(self.route_history),
            "visited_customers_count": len(self.visited_customers),
            "all_customers_served": all_customers_served,
            "total_distance": round(self.total_distance, 4),
            "total_energy": round(self.total_energy, 4),
            "action_mask": self.get_action_mask(),
        }

        return self._get_observation(), float(reward), terminated, truncated, info
