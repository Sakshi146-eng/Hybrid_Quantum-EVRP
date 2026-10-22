"""
Reinforcement Learning Agent for Subproblem Selection and QAOA Parameter Adaptation.

Implements:
1. Feature extraction for EV routing subproblem candidates.
2. Neural Subproblem Priority Scorer: ranks subproblems by expected improvement.
3. Adaptive Parameter Policy: selects circuit depth p, penalty weight, and shot budget.
4. Online policy update with EV cost reduction feedback.
"""
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.evrp.instance import EVRPInstance


class SubproblemPriorityNet(nn.Module):
    """
    Actor network predicting subproblem promise score and adaptive QAOA execution parameters.
    """

    def __init__(self, input_dim: int = 7):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        # Priority score head (scalar value)
        self.score_head = nn.Linear(16, 1)

        # Parameter decision heads
        # 1. Depth p (p=1 vs p=2)
        self.depth_head = nn.Linear(16, 2)
        # 2. Penalty multiplier lambda (1.5, 2.0, 3.0)
        self.penalty_head = nn.Linear(16, 3)
        # 3. Shots budget (256, 512, 1024)
        self.shots_head = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        score = self.score_head(features).squeeze(-1)
        depth_logits = self.depth_head(features)
        penalty_logits = self.penalty_head(features)
        shots_logits = self.shots_head(features)
        return score, depth_logits, penalty_logits, shots_logits


class RLSubproblemAgent:
    """
    RL Agent guiding WHERE (subproblem window) and HOW (circuit depth, penalty, shots)
    QAOA should be invoked in hybrid EVRP optimization.
    """

    def __init__(
        self,
        lr: float = 0.002,
        gamma: float = 0.95,
        seed: Optional[int] = 42,
        device: Optional[str] = None,
    ):
        self.seed = seed or 42
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.network = SubproblemPriorityNet(input_dim=7).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma

        # Memory buffer for online updates
        self.trajectory_buffer: List[Dict[str, Any]] = []
        self.visit_counts: Dict[str, int] = {}

    def extract_subproblem_features(
        self,
        instance: EVRPInstance,
        route: List[int],
        window_indices: List[int],
        entry_load: float,
        entry_battery: float,
    ) -> np.ndarray:
        """
        Extracts normalized 7-dimensional feature vector for a candidate subproblem.
        Features:
          0: Subproblem size normalized (m / 4.0)
          1: Detour factor (direct entry->exit dist / perimeter of subproblem)
          2: Edge length variance / mean (structural irregularity)
          3: Load ratio (current vehicle load / vehicle capacity)
          4: Battery stress (1.0 - current SoC / battery capacity)
          5: Proximity to nearest charging station
          6: Previous visit penalty (to prevent redundant evaluations)
        """
        entry_node = route[window_indices[0] - 1]
        exit_node = route[window_indices[-1] + 1]
        cust_nodes = [route[i] for i in window_indices]
        m = len(cust_nodes)
        D = instance.distance_matrix

        # 1. Size
        f_size = float(m) / 4.0

        # 2. Direct distance vs perimeter
        direct_dist = float(D[entry_node, exit_node])
        perimeter = float(D[entry_node, cust_nodes[0]])
        for idx in range(m - 1):
            perimeter += float(D[cust_nodes[idx], cust_nodes[idx + 1]])
        perimeter += float(D[cust_nodes[-1], exit_node])

        f_detour = (perimeter - direct_dist) / max(1.0, direct_dist)
        f_detour = float(np.clip(f_detour, 0.0, 5.0) / 5.0)

        # 3. Edge length variance
        edge_lens = [float(D[entry_node, cust_nodes[0]])]
        for idx in range(m - 1):
            edge_lens.append(float(D[cust_nodes[idx], cust_nodes[idx + 1]]))
        edge_lens.append(float(D[cust_nodes[-1], exit_node]))
        f_variance = float(np.std(edge_lens) / (np.mean(edge_lens) + 1e-6))
        f_variance = float(np.clip(f_variance, 0.0, 3.0) / 3.0)

        # 4. Load ratio
        veh_cap = float(instance.capacities[0]) if instance.capacities else 100.0
        f_load = float(np.clip(entry_load / max(1.0, veh_cap), 0.0, 1.0))

        # 5. Battery stress
        batt_cap = float(instance.battery_capacity)
        f_battery = float(np.clip(1.0 - (entry_battery / max(1.0, batt_cap)), 0.0, 1.0))

        # 6. Charging station distance
        if instance.charging_stations:
            min_cs = min(float(D[c, s]) for c in cust_nodes for s in instance.charging_stations)
            f_cs = float(np.clip(min_cs / 50.0, 0.0, 1.0))
        else:
            f_cs = 0.5

        # 7. Visit count penalty
        sub_key = f"{entry_node}_{'-'.join(str(c) for c in sorted(cust_nodes))}_{exit_node}"
        visits = self.visit_counts.get(sub_key, 0)
        f_visits = float(min(1.0, visits / 3.0))

        return np.array([f_size, f_detour, f_variance, f_load, f_battery, f_cs, f_visits], dtype=np.float32)

    def rank_candidate_subproblems(
        self,
        candidate_items: List[Dict[str, Any]],
        deterministic: bool = False,
    ) -> List[Tuple[int, Dict[str, Any], float]]:
        """
        Scores all candidate subproblems and returns them ranked by expected optimization yield.
        Returns:
            List of (candidate_index, chosen_params, predicted_score)
        """
        if not candidate_items:
            return []

        features_list = [item["features"] for item in candidate_items]
        features_t = torch.FloatTensor(np.array(features_list)).to(self.device)

        self.network.eval()
        with torch.no_grad():
            scores_t, depth_log, pen_log, shot_log = self.network(features_t)
            scores = scores_t.cpu().numpy()

            depth_probs = torch.softmax(depth_log, dim=-1).cpu().numpy()
            pen_probs = torch.softmax(pen_log, dim=-1).cpu().numpy()
            shot_probs = torch.softmax(shot_log, dim=-1).cpu().numpy()

        ranked: List[Tuple[int, Dict[str, Any], float]] = []

        depth_options = [1, 2]
        penalty_options = [1.5, 2.0, 3.0]
        shot_options = [256, 512, 1024]

        for i, item in enumerate(candidate_items):
            base_score = float(scores[i])

            if deterministic:
                d_idx = int(np.argmax(depth_probs[i]))
                p_idx = int(np.argmax(pen_probs[i]))
                s_idx = int(np.argmax(shot_probs[i]))
            else:
                d_idx = int(np.random.choice(len(depth_options), p=depth_probs[i]))
                p_idx = int(np.random.choice(len(penalty_options), p=pen_probs[i]))
                s_idx = int(np.random.choice(len(shot_options), p=shot_probs[i]))

            chosen_params = {
                "p_depth": depth_options[d_idx],
                "penalty_multiplier": penalty_options[p_idx],
                "shots": shot_options[s_idx],
                "depth_idx": d_idx,
                "pen_idx": p_idx,
                "shot_idx": s_idx,
            }
            ranked.append((i, chosen_params, base_score))

        # Sort descending by priority score
        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked

    def record_step_result(
        self,
        features: np.ndarray,
        params: Dict[str, Any],
        improvement_cost: float,
        prior_cost: float,
        runtime_sec: float,
        subproblem_key: str,
    ) -> None:
        """Records experience tuple for online policy gradient update."""
        self.visit_counts[subproblem_key] = self.visit_counts.get(subproblem_key, 0) + 1

        rel_gain = (improvement_cost / max(1.0, prior_cost)) * 100.0
        # Reward combines relative cost improvement and computational efficiency penalty
        reward = float(rel_gain - 0.5 * runtime_sec)

        self.trajectory_buffer.append({
            "features": features,
            "depth_idx": params.get("depth_idx", 0),
            "pen_idx": params.get("pen_idx", 1),
            "shot_idx": params.get("shot_idx", 1),
            "reward": reward,
        })

    def train_step(self) -> Optional[float]:
        """Executes a policy gradient update step if buffer has sufficient experiences."""
        if len(self.trajectory_buffer) < 4:
            return None

        self.network.train()
        feats = torch.FloatTensor(np.array([item["features"] for item in self.trajectory_buffer])).to(self.device)
        rewards = torch.FloatTensor([item["reward"] for item in self.trajectory_buffer]).to(self.device)

        d_targets = torch.LongTensor([item["depth_idx"] for item in self.trajectory_buffer]).to(self.device)
        p_targets = torch.LongTensor([item["pen_idx"] for item in self.trajectory_buffer]).to(self.device)
        s_targets = torch.LongTensor([item["shot_idx"] for item in self.trajectory_buffer]).to(self.device)

        # Normalize rewards
        if len(rewards) > 1 and torch.std(rewards) > 1e-6:
            advantages = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-6)
        else:
            advantages = rewards

        scores, depth_log, pen_log, shot_log = self.network(feats)

        # Priority score regression loss (predicting true advantage)
        score_loss = nn.MSELoss()(scores, advantages)

        # Policy losses
        d_loss = nn.CrossEntropyLoss(reduction="none")(depth_log, d_targets)
        p_loss = nn.CrossEntropyLoss(reduction="none")(pen_log, p_targets)
        s_loss = nn.CrossEntropyLoss(reduction="none")(shot_log, s_targets)

        policy_loss = (d_loss * advantages.detach() + p_loss * advantages.detach() + s_loss * advantages.detach()).mean()
        total_loss = score_loss + 0.5 * policy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.trajectory_buffer.clear()
        return float(total_loss.item())

    def save_checkpoint(self, path: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "visit_counts": self.visit_counts,
        }, p)

    def load_checkpoint(self, path: Any) -> bool:
        p = Path(path)
        if not p.is_file():
            return False
        try:
            ckpt = torch.load(p, map_location=self.device)
            self.network.load_state_dict(ckpt["model_state_dict"])
            self.visit_counts = ckpt.get("visit_counts", {})
            return True
        except Exception:
            return False
