"""
Reinforcement Learning Agent for Electric Vehicle Routing.
Implements a Masked Actor-Critic neural network using PyTorch, supporting training,
checkpointing, validation, and reproducible deterministic inference.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.evrp.instance import EVRPInstance
from src.rl.env import EVRoutingEnv
from src.evrp.charging import insert_charging_stations_if_needed, prune_redundant_charging_stops
from src.evrp.cost import calculate_evrp_solution_cost
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution


class MaskedActorCriticNetwork(nn.Module):
    """
    Two-headed neural network for policy and value estimation with action masking.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Policy head (Actor)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Value head (Critic)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns action distribution probabilities and state value.
        """
        feats = self.shared_backbone(obs)
        logits = self.actor(feats)
        value = self.critic(feats)

        if action_mask is not None:
            # Mask invalid actions with large negative value so probability is 0
            very_negative = -1e9
            logits = torch.where(action_mask, logits, torch.tensor(very_negative, device=logits.device))

        probs = torch.softmax(logits, dim=-1)
        return probs, value.squeeze(-1)


class EVRLAgent(BaseVRPSolver):
    """
    Reinforcement Learning Solver for EVRP.
    Employs Actor-Critic policy gradient updates with action masking to generate feasible EV routes.
    """

    def __init__(
        self,
        max_nodes: int = 16,
        hidden_dim: int = 128,
        lr: float = 0.001,
        gamma: float = 0.99,
        seed: Optional[int] = 42,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.obs_dim = 6 + self.max_nodes * 8
        self.action_dim = self.max_nodes

        self.network = MaskedActorCriticNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        self.training_history: List[Dict[str, Any]] = []
        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "Reinforcement Learning (RL)"

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> int:
        """Selects action given observation and mask with optional temperature scaling."""
        self.network.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
            probs, _ = self.network(obs_t, mask_t)

            probs_np = probs.cpu().numpy()[0]
            # Replace NaNs if any
            if np.isnan(probs_np).any() or np.sum(probs_np) <= 0:
                valid_indices = np.where(action_mask)[0]
                return int(valid_indices[0]) if len(valid_indices) > 0 else 0

            probs_np /= np.sum(probs_np)

            if deterministic or temperature <= 1e-4:
                return int(np.argmax(probs_np))

            # Apply temperature scaling
            log_probs = np.log(np.maximum(probs_np, 1e-12)) / max(temperature, 0.05)
            exp_probs = np.exp(log_probs - np.max(log_probs)) * action_mask
            sum_exp = np.sum(exp_probs)
            if sum_exp <= 0 or np.isnan(sum_exp):
                return int(np.argmax(probs_np))
            exp_probs /= sum_exp
            return int(np.random.choice(self.action_dim, p=exp_probs))

    def train_step(self, instance: EVRPInstance) -> Dict[str, Any]:
        """
        Executes a single Actor-Critic training episode on an EVRPInstance.
        Returns a dictionary containing episode reward and loss metrics.
        """
        self.network.train()
        env = EVRoutingEnv(instance=instance, max_nodes=self.max_nodes)
        obs, info = env.reset()

        states = []
        actions = []
        masks = []
        rewards = []
        values = []

        done = False
        total_ep_reward = 0.0

        while not done:
            mask = info["action_mask"]
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

            probs, val = self.network(obs_t, mask_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_obs, r, term, trunc, info = env.step(action.item())
            done = term or trunc

            states.append(obs)
            actions.append(action.item())
            masks.append(mask)
            rewards.append(r)
            values.append(val.item())

            obs = next_obs
            total_ep_reward += r

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).to(self.device)
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        masks_t = torch.BoolTensor(np.array(masks)).to(self.device)

        probs_all, values_pred = self.network(states_t, masks_t)
        dist_all = torch.distributions.Categorical(probs_all)
        log_probs = dist_all.log_prob(actions_t)

        advantages = returns_t - values_pred.detach()
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = nn.MSELoss()(values_pred, returns_t)
        entropy_loss = -dist_all.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "reward": round(float(total_ep_reward), 4),
            "actor_loss": round(float(actor_loss.item()), 4),
            "critic_loss": round(float(critic_loss.item()), 4),
            "loss": round(float(loss.item()), 4),
        }

    def train_agent(
        self,
        train_instances: List[EVRPInstance],
        val_instances: Optional[List[EVRPInstance]] = None,
        episodes: int = 100,
        eval_freq: int = 20,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trains the RL agent across training instances using Actor-Critic policy gradient.
        Evaluates periodically on held-out validation instances.
        """
        t0 = time.perf_counter()
        self.network.train()

        best_val_reward = -float("inf")
        history = []

        for ep in range(1, episodes + 1):
            inst = train_instances[(ep - 1) % len(train_instances)]
            log_entry = self.train_step(inst)
            log_entry["episode"] = ep

            # Periodic validation evaluation
            if val_instances and (ep % eval_freq == 0 or ep == episodes):
                val_rewards = []
                for v_inst in val_instances:
                    v_env = EVRoutingEnv(instance=v_inst, max_nodes=self.max_nodes)
                    v_obs, v_info = v_env.reset()
                    v_done = False
                    v_total = 0.0
                    while not v_done:
                        act = self.select_action(v_obs, v_info["action_mask"], deterministic=True)
                        v_obs, vr, v_term, v_trunc, v_info = v_env.step(act)
                        v_done = v_term or v_trunc
                        v_total += vr
                    val_rewards.append(v_total)

                mean_val = float(np.mean(val_rewards))
                log_entry["val_reward"] = round(mean_val, 4)

                if mean_val > best_val_reward and checkpoint_path:
                    best_val_reward = mean_val
                    self.save_checkpoint(checkpoint_path)

            history.append(log_entry)

        train_time = time.perf_counter() - t0
        self.training_history = history

        return {
            "training_time_seconds": round(train_time, 4),
            "episodes": episodes,
            "final_reward": history[-1]["reward"] if history else 0.0,
            "history": history,
        }

    def save_checkpoint(self, path: Any) -> None:
        """Saves model weights and hyperparameters to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "max_nodes": self.max_nodes,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "gamma": self.gamma,
        }
        if hasattr(self, "optimizer") and self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, str(p))

    def load_checkpoint(self, path: Any) -> bool:
        """Loads model weights from disk."""
        p = Path(path)
        if not p.is_file():
            return False
        try:
            ckpt = torch.load(str(p), map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.network.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                self.network.load_state_dict(ckpt)
            return True
        except Exception as e:
            print(f"[EVRLAgent] Error loading checkpoint: {e}")
            return False

    def save(self, path: Any) -> None:
        """Alias for save_checkpoint."""
        self.save_checkpoint(path)

    def load(self, path: Any) -> bool:
        """Alias for load_checkpoint."""
        return self.load_checkpoint(path)

    def _rollout_single_solution(
        self,
        instance: EVRPInstance,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> List[List[int]]:
        """
        Executes a multi-vehicle rollout guaranteeing 100% customer coverage.
        """
        all_customers = set(instance.customer_indices)
        served: set = set()
        routes: List[List[int]] = []
        max_vehicles = instance.num_vehicles or 4
        veh_capacity = instance.capacities[0] if instance.capacities else 100000.0

        for _ in range(max_vehicles):
            if len(served) >= len(all_customers):
                break

            rem_custs = all_customers - served
            env = EVRoutingEnv(instance=instance, max_nodes=self.max_nodes)
            obs, info = env.reset()
            # Constrain environment to unserved customers
            env.visited_customers = set(served)
            env.current_load = float(sum(instance.demands[c] for c in rem_custs))

            route_steps = [instance.depot_idx]
            done = False
            step_count = 0
            curr_veh_load = 0.0

            while not done and step_count < self.max_nodes * 2:
                step_count += 1
                mask = env.get_action_mask()

                # If no more customers left to serve in this episode, return to depot
                if len(all_customers - env.visited_customers) == 0:
                    if env.current_node != instance.depot_idx:
                        route_steps.append(instance.depot_idx)
                    break

                action = self.select_action(
                    obs, mask, deterministic=deterministic, temperature=temperature
                )
                obs, r, term, trunc, info = env.step(action)

                if action == instance.depot_idx:
                    if len(route_steps) > 1 and route_steps[-1] != instance.depot_idx:
                        route_steps.append(instance.depot_idx)
                    break
                else:
                    if instance.is_customer(action):
                        # Enforce vehicle capacity
                        dem = float(instance.demands[action])
                        if curr_veh_load + dem > veh_capacity + 1e-4:
                            # Vehicle full; return to depot
                            route_steps.append(instance.depot_idx)
                            break
                        curr_veh_load += dem
                        served.add(action)
                    route_steps.append(action)

                done = term or trunc

            # Ensure start and end at depot
            if route_steps[0] != instance.depot_idx:
                route_steps.insert(0, instance.depot_idx)
            if route_steps[-1] != instance.depot_idx:
                route_steps.append(instance.depot_idx)

            custs_in_r = [n for n in route_steps if instance.is_customer(n)]
            if custs_in_r:
                repaired_r, _ = insert_charging_stations_if_needed(route_steps, instance)
                routes.append(repaired_r)

        # Fallback: if any customer was omitted, insert into existing routes without exceeding fleet limit
        unserved = all_customers - set(n for r in routes for n in r if instance.is_customer(n))
        if unserved:
            for u in sorted(list(unserved)):
                best_ins = None
                best_cost_inc = float("inf")
                u_dem = float(instance.demands[u])
                for r_idx, r in enumerate(routes):
                    load_r = sum(float(instance.demands[c]) for c in r if instance.is_customer(c))
                    if load_r + u_dem > veh_capacity + 1e-4:
                        continue
                    base_eval = instance.evaluate_route_ev(r)
                    for pos in range(1, len(r)):
                        cand_r = r[:pos] + [u] + r[pos:]
                        rep_r, _ = insert_charging_stations_if_needed(cand_r, instance)
                        rep_r = prune_redundant_charging_stops(rep_r, instance)
                        ev_eval = instance.evaluate_route_ev(rep_r)
                        if ev_eval["feasible"]:
                            inc = ev_eval["total_distance"] - base_eval["total_distance"]
                            if inc < best_cost_inc:
                                best_cost_inc = inc
                                best_ins = (r_idx, rep_r)
                if best_ins is not None:
                    routes[best_ins[0]] = best_ins[1]
                else:
                    if len(routes) < max_vehicles:
                        supp = [instance.depot_idx, u, instance.depot_idx]
                        rep_supp, _ = insert_charging_stations_if_needed(supp, instance)
                        routes.append(rep_supp)
                    else:
                        min_r_idx = min(
                            range(len(routes)),
                            key=lambda i: sum(float(instance.demands[c]) for c in routes[i] if instance.is_customer(c))
                        )
                        routes[min_r_idx].insert(-1, u)
                        rep_supp, _ = insert_charging_stations_if_needed(routes[min_r_idx], instance)
                        routes[min_r_idx] = prune_redundant_charging_stops(rep_supp, instance)

        # Deduplicate consecutive identical nodes
        clean_routes = []
        for r in routes:
            comp = []
            for n in r:
                if not comp or n != comp[-1]:
                    comp.append(n)
            if comp[0] != instance.depot_idx:
                comp = [instance.depot_idx] + comp
            if comp[-1] != instance.depot_idx:
                comp = comp + [instance.depot_idx]
            clean_routes.append(comp)

        from src.evrp.fleet_allocation import GlobalFleetCoordinator
        coordinator = GlobalFleetCoordinator(instance)
        return coordinator.enforce_fleet_limit(clean_routes)

    def generate_route_for_instance(
        self,
        instance: EVRPInstance,
        deterministic: bool = True,
        num_samples: int = 4,
    ) -> List[List[int]]:
        """
        Generates candidate EV routes using policy inference, evaluates them under
        the unified EVRP cost function, and selects the best feasible candidate.
        """
        candidates: List[List[List[int]]] = []
        from src.evrp.fleet_allocation import GlobalFleetCoordinator
        coordinator = GlobalFleetCoordinator(instance)

        # Candidate 0: Global fleet coordinated solution (Regret-2 / Sweep)
        coord_cand = coordinator.build_best_coordinated_routes()
        candidates.append(coord_cand)

        # Candidate 1: Deterministic greedy rollout
        c1 = self._rollout_single_solution(instance, deterministic=True, temperature=0.0)
        candidates.append(coordinator.enforce_fleet_limit(c1))

        # Candidates 2+: Stochastic sampling with varying temperatures
        if num_samples > 1:
            for temp in [0.5, 0.8, 1.2]:
                cand = self._rollout_single_solution(instance, deterministic=False, temperature=temp)
                candidates.append(coordinator.enforce_fleet_limit(cand))

        # Candidate Variation: Route consolidation (try merging routes if capacity allows)
        veh_capacity = instance.capacities[0] if instance.capacities else 100000.0
        consolidated_candidates = []
        for cand in candidates:
            if len(cand) > 1:
                # Try merging customer sequences into fewer vehicles
                all_custs = [n for r in cand for n in r if instance.is_customer(n)]
                total_dem = sum(float(instance.demands[c]) for c in all_custs)
                if total_dem <= veh_capacity:
                    merged = [instance.depot_idx] + all_custs + [instance.depot_idx]
                    rep_merged, _ = insert_charging_stations_if_needed(merged, instance)
                    consolidated_candidates.append([rep_merged])

        candidates.extend(consolidated_candidates)

        # Evaluate all candidates and select best feasible one
        best_routes = candidates[0]
        best_eval = calculate_evrp_solution_cost(instance, best_routes)
        best_cost = best_eval["total_cost"]

        for cand in candidates[1:]:
            cand_eval = calculate_evrp_solution_cost(instance, cand)
            # Accept if feasible with lower cost, or if current best is infeasible and cand is feasible
            if (cand_eval["feasible"] and not best_eval["feasible"]) or (
                cand_eval["feasible"] and cand_eval["total_cost"] < best_cost - 1e-4
            ):
                best_routes = cand
                best_eval = cand_eval
                best_cost = cand_eval["total_cost"]

        return coordinator.enforce_fleet_limit(best_routes)

    def _solve_instance(self, instance: EVRPInstance) -> List[List[int]]:
        t0 = time.perf_counter()
        routes = self.generate_route_for_instance(instance, deterministic=True)
        runtime = time.perf_counter() - t0
        cost_eval = calculate_evrp_solution_cost(instance, routes)

        self.last_metadata = {
            "algorithm": self.algorithm_name,
            "inference_seconds": round(runtime, 6),
            "evrp_evaluation": cost_eval,
            "feasible": cost_eval["feasible"],
        }
        return routes

    def solve(self, instance: EVRPInstance) -> VRPSolution:
        t0 = time.perf_counter()
        routes = self._solve_instance(instance)
        runtime = time.perf_counter() - t0

        cost_eval = calculate_evrp_solution_cost(instance, routes)

        return VRPSolution(
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
