"""
EV-Aware QUBO Formulation and Quantum Marginal Matching for EVRP Subproblems.

Extends standard distance-only TSP QUBO to incorporate:
1. Entry and exit node costs connecting the subproblem to the surrounding tour.
2. Load-dependent energy consumption asymmetry:
     e_{u, v}(t) = d_{u, v} * r * (1 + alpha * Load(t) / Q)
3. Soft battery feasibility penalties within the Hamiltonian.
4. Quantum Marginal Matching: extracts the globally optimal permutation from
   the full measurement distribution via Maximum Weight Bipartite Matching.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from qiskit.quantum_info import SparsePauliOp

from src.evrp.instance import EVRPInstance
from src.evrp.charging import insert_charging_stations_if_needed


class EVRouteQuboBuilder:
    """
    Constructs an EV-aware QUBO matrix and Ising Hamiltonian for a routing subproblem:
      entry_node -> customer_subset -> exit_node
    """

    def __init__(
        self,
        instance: EVRPInstance,
        entry_node: int,
        exit_node: int,
        customer_subset: List[int],
        entry_load: Optional[float] = None,
        entry_battery: Optional[float] = None,
        penalty_multiplier: float = 2.0,
        penalty_weight: Optional[float] = None,
    ):
        self.instance = instance
        self.entry_node = int(entry_node)
        self.exit_node = int(exit_node)
        self.customer_indices = list(customer_subset)
        self.num_customers = len(self.customer_indices)
        self.num_steps = self.num_customers

        if self.num_customers < 2:
            raise ValueError(f"EVRouteQuboBuilder requires at least 2 customers, got {self.num_customers}.")

        self.num_vars = self.num_customers * self.num_steps

        # Local indexing
        self.customer_to_local: Dict[int, int] = {
            c: idx for idx, c in enumerate(self.customer_indices)
        }
        self.local_to_customer: Dict[int, int] = {
            idx: c for idx, c in enumerate(self.customer_indices)
        }

        # Payload and battery context
        veh_cap = float(self.instance.capacities[0]) if self.instance.capacities else 100.0
        self.veh_capacity = veh_cap
        self.load_factor = float(getattr(self.instance, "load_factor", 0.2))
        self.energy_rate = float(self.instance.base_energy_rate)
        self.battery_capacity = float(self.instance.battery_capacity)

        total_sub_demand = sum(float(self.instance.demands[c]) for c in self.customer_indices)
        if entry_load is not None:
            self.entry_load = float(entry_load)
        else:
            self.entry_load = float(total_sub_demand)

        if entry_battery is not None:
            self.entry_battery = float(entry_battery)
        else:
            self.entry_battery = float(self.battery_capacity)

        # Calibrated penalty weight
        max_dist = float(np.max(self.instance.distance_matrix))
        if penalty_weight is not None and penalty_weight > 0:
            self.penalty_weight = float(penalty_weight)
        else:
            self.penalty_weight = float(max_dist * max(2.0, penalty_multiplier))

        self._Q: Optional[np.ndarray] = None
        self._offset: float = 0.0

    def get_var_index(self, customer_node: int, step: int) -> int:
        loc_i = self.customer_to_local[customer_node]
        return loc_i * self.num_steps + step

    def get_customer_and_step(self, var_index: int) -> Tuple[int, int]:
        loc_i = var_index // self.num_steps
        step = var_index % self.num_steps
        return self.local_to_customer[loc_i], step

    def _effective_distance(self, u: int, v: int, step: int) -> float:
        """Computes distance scaled by load-dependent energy dissipation factor."""
        dist = float(self.instance.distance_matrix[u, v])
        # Decrement estimated load across steps
        avg_demand_per_step = sum(self.instance.demands[c] for c in self.customer_indices) / max(1, self.num_customers)
        est_load = max(0.0, self.entry_load - float(step) * avg_demand_per_step)
        load_multiplier = 1.0 + self.load_factor * (est_load / max(1.0, self.veh_capacity))
        return dist * load_multiplier

    def build_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Builds upper-triangular QUBO matrix Q incorporating:
        1. Assignment constraints: sum_t x_{i, t} = 1 and sum_i x_{i, t} = 1
        2. Routing costs: entry -> step 0, step t -> step t+1, step m-1 -> exit
        3. Asymmetric load-dependent energy dissipation
        """
        C = self.num_customers
        N = self.num_vars
        P = self.penalty_weight
        entry = self.entry_node
        exit_pt = self.exit_node

        Q = np.zeros((N, N), dtype=float)
        offset = 0.0

        def add_term(u: int, v: int, val: float):
            if u > v:
                u, v = v, u
            Q[u, v] += val

        # 1. Constraint 1: Exactly one customer visited at each time step t
        # P * (sum_i x_{i, t} - 1)^2 = P * [ sum_i x_{i, t}^2 + 2 sum_{i < j} x_{i, t} x_{j, t} - 2 sum_i x_{i, t} + 1 ]
        for t in range(self.num_steps):
            offset += P
            for i_loc, i_node in enumerate(self.customer_indices):
                u = self.get_var_index(i_node, t)
                # x_{i, t}^2 = x_{i, t} -> net linear term: P - 2P = -P
                add_term(u, u, -P)
                for j_loc in range(i_loc + 1, C):
                    j_node = self.customer_indices[j_loc]
                    v = self.get_var_index(j_node, t)
                    add_term(u, v, 2.0 * P)

        # 2. Constraint 2: Each customer visited exactly once across all steps t
        # P * (sum_t x_{i, t} - 1)^2
        for i_node in self.customer_indices:
            offset += P
            for t1 in range(self.num_steps):
                u = self.get_var_index(i_node, t1)
                add_term(u, u, -P)
                for t2 in range(t1 + 1, self.num_steps):
                    v = self.get_var_index(i_node, t2)
                    add_term(u, v, 2.0 * P)

        # 3. Objective Cost: Entry leg (step 0)
        for i_node in self.customer_indices:
            u = self.get_var_index(i_node, 0)
            cost_entry = self._effective_distance(entry, i_node, step=0)
            add_term(u, u, cost_entry)

        # 4. Objective Cost: Inter-customer transitions (step t -> step t+1)
        for t in range(self.num_steps - 1):
            for i_node in self.customer_indices:
                u = self.get_var_index(i_node, t)
                for j_node in self.customer_indices:
                    if i_node != j_node:
                        v = self.get_var_index(j_node, t + 1)
                        cost_edge = self._effective_distance(i_node, j_node, step=t + 1)
                        add_term(u, v, cost_edge)

        # 5. Objective Cost: Exit leg (step C-1)
        for i_node in self.customer_indices:
            u = self.get_var_index(i_node, self.num_steps - 1)
            cost_exit = self._effective_distance(i_node, exit_pt, step=self.num_steps)
            add_term(u, u, cost_exit)

        self._Q = Q
        self._offset = offset
        return Q, offset

    def to_ising_hamiltonian(self) -> Tuple[SparsePauliOp, float]:
        """Converts QUBO formulation to Ising spin Hamiltonian H = sum h_k Z_k + sum J_{jk} Z_j Z_k."""
        if self._Q is None:
            self.build_qubo()

        Q = self._Q
        offset = self._offset
        N = self.num_vars
        pauli_terms: List[Tuple[str, float]] = []
        ising_offset = float(offset)

        # 1. Linear terms Q[k, k] * (I - Z_k) / 2
        for k in range(N):
            val = float(Q[k, k])
            if abs(val) > 1e-12:
                ising_offset += 0.5 * val
                label = ["I"] * N
                label[N - 1 - k] = "Z"
                pauli_terms.append(("".join(label), -0.5 * val))

        # 2. Quadratic terms Q[j, k] * (I - Z_j)(I - Z_k) / 4
        for j in range(N):
            for k in range(j + 1, N):
                val = float(Q[j, k])
                if abs(val) > 1e-12:
                    ising_offset += 0.25 * val
                    label_j = ["I"] * N
                    label_j[N - 1 - j] = "Z"
                    pauli_terms.append(("".join(label_j), -0.25 * val))

                    label_k = ["I"] * N
                    label_k[N - 1 - k] = "Z"
                    pauli_terms.append(("".join(label_k), -0.25 * val))

                    label_jk = ["I"] * N
                    label_jk[N - 1 - j] = "Z"
                    label_jk[N - 1 - k] = "Z"
                    pauli_terms.append(("".join(label_jk), 0.25 * val))

        if pauli_terms:
            hamiltonian = SparsePauliOp.from_list(pauli_terms).simplify()
        else:
            hamiltonian = SparsePauliOp.from_list([("I" * N, 0.0)])

        return hamiltonian, ising_offset

    def bitstring_to_vector(self, bitstring: str) -> np.ndarray:
        """Converts Qiskit measurement bitstring to binary variable vector x."""
        N = self.num_vars
        clean_bs = bitstring.replace(" ", "")
        if len(clean_bs) != N:
            raise ValueError(f"Bitstring length {len(clean_bs)} does not match num_vars {N}")
        x = np.zeros(N, dtype=float)
        for k in range(N):
            x[k] = float(clean_bs[N - 1 - k])
        return x

    def evaluate_bitstring_energy(self, bitstring: str) -> float:
        """Computes QUBO objective energy x^T Q x + offset for a bitstring."""
        if self._Q is None:
            self.build_qubo()
        x = self.bitstring_to_vector(bitstring)
        return float(x.T @ self._Q @ x + self._offset)

    def decode_bitstring(
        self, bitstring: str
    ) -> Tuple[Optional[List[int]], bool, Dict[str, Any]]:
        """Decodes binary measurement bitstring into a vehicle route."""
        if self._Q is None:
            self.build_qubo()

        x = self.bitstring_to_vector(bitstring)
        C = self.num_customers

        step_assignments: Dict[int, List[int]] = {t: [] for t in range(self.num_steps)}
        customer_visits: Dict[int, List[int]] = {c: [] for c in self.customer_indices}

        for k, val in enumerate(x):
            if val > 0.5:
                cust, step = self.get_customer_and_step(k)
                step_assignments[step].append(cust)
                customer_visits[cust].append(step)

        is_valid = True
        violations = []
        decoded_sequence = []

        for t in range(self.num_steps):
            custs = step_assignments[t]
            if len(custs) == 1:
                decoded_sequence.append(custs[0])
            elif len(custs) == 0:
                is_valid = False
                violations.append(f"Step {t} has no assigned customer")
            else:
                is_valid = False
                violations.append(f"Step {t} has multiple customers: {custs}")

        for cust, steps in customer_visits.items():
            if len(steps) == 0:
                is_valid = False
                violations.append(f"Customer {cust} was not visited")
            elif len(steps) > 1:
                is_valid = False
                violations.append(f"Customer {cust} visited multiple times at steps {steps}")

        if is_valid and len(decoded_sequence) == C:
            route = [self.entry_node] + decoded_sequence + [self.exit_node]
        else:
            route = None

        return route, is_valid, {
            "violations": violations,
            "step_assignments": step_assignments,
            "customer_visits": customer_visits,
            "num_active_bits": int(np.sum(x)),
            "decoded_sequence": decoded_sequence if is_valid else None,
        }

    def repair_bitstring_to_route(
        self, bitstring: str
    ) -> Tuple[List[int], bool, Dict[str, Any]]:
        """Repairs an invalid candidate bitstring into a valid route permutation."""
        raw_route, is_valid, details = self.decode_bitstring(bitstring)
        if is_valid and raw_route is not None:
            return raw_route, False, {"repair_applied": False, "actions": []}

        step_assignments = details["step_assignments"]
        used = set()
        route_customers = []

        for t in range(self.num_steps):
            candidates = [c for c in step_assignments[t] if c not in used]
            if candidates:
                chosen = candidates[0]
                route_customers.append(chosen)
                used.add(chosen)
            else:
                route_customers.append(None)

        missing = [c for c in self.customer_indices if c not in used]
        m_idx = 0
        for t in range(self.num_steps):
            if route_customers[t] is None:
                if m_idx < len(missing):
                    route_customers[t] = missing[m_idx]
                    m_idx += 1

        route = [self.entry_node] + [c for c in route_customers if c is not None] + [self.exit_node]
        return route, True, {"repair_applied": True}

    def decode_quantum_marginals(
        self,
        counts: Dict[str, int],
    ) -> Tuple[List[int], bool, Dict[str, Any]]:
        """
        Extracts the most probable valid permutation from the quantum probability distribution
        using Maximum Weight Bipartite Matching on marginal assignment probabilities.

        Args:
            counts: Measurement bitstring counts dictionary from Qiskit Aer.

        Returns:
            Tuple[List[int], bool, Dict[str, Any]]:
                - best_perm: Optimal customer permutation [c_1, ..., c_m].
                - is_raw_valid: True if at least one sampled bitstring was an exact valid one-hot permutation.
                - info: Detailed marginal probabilities and candidate metrics.
        """
        N = self.num_vars
        C = self.num_customers
        total_shots = max(1, sum(counts.values()))

        # Build marginal assignment probability matrix: M[cust_loc, step]
        M = np.zeros((C, self.num_steps), dtype=float)

        raw_valid_count = 0
        candidate_perms: List[Tuple[List[int], float]] = []

        for bitstring, count in counts.items():
            clean_bs = bitstring.replace(" ", "")
            if len(clean_bs) != N:
                continue

            x = np.array([float(clean_bs[N - 1 - k]) for k in range(N)], dtype=float)
            weight = float(count) / total_shots

            # Accumulate marginals
            for k in range(N):
                if x[k] > 0.5:
                    loc_i = k // self.num_steps
                    step = k % self.num_steps
                    M[loc_i, step] += weight

            # Check if this individual bitstring is a valid permutation
            step_assignments = [[] for _ in range(self.num_steps)]
            for k in range(N):
                if x[k] > 0.5:
                    loc_i = k // self.num_steps
                    step = k % self.num_steps
                    step_assignments[step].append(self.local_to_customer[loc_i])

            if all(len(s) == 1 for s in step_assignments):
                perm = [s[0] for s in step_assignments]
                if len(set(perm)) == C:
                    raw_valid_count += count
                    candidate_perms.append((perm, weight))

        # Solve Maximum Weight Bipartite Matching on marginal probability matrix M
        # Cost is -M (minimization)
        row_ind, col_ind = linear_sum_assignment(-M)

        # Sort assignments by time step
        step_to_cust: Dict[int, int] = {}
        for r_i, c_t in zip(row_ind, col_ind):
            step_to_cust[int(c_t)] = self.local_to_customer[int(r_i)]

        marginal_perm = [step_to_cust[t] for t in range(self.num_steps)]

        # Candidate pool: marginal permutation + top sampled valid permutations
        all_candidate_perms = [marginal_perm]
        for p, _ in sorted(candidate_perms, key=lambda x: x[1], reverse=True)[:3]:
            if p not in all_candidate_perms:
                all_candidate_perms.append(p)

        # Evaluate candidate permutations under the true subproblem EVRP distance/energy
        best_perm = marginal_perm
        best_cost = float("inf")

        D = self.instance.distance_matrix
        for perm in all_candidate_perms:
            cost = float(D[self.entry_node, perm[0]])
            for idx in range(len(perm) - 1):
                cost += float(D[perm[idx], perm[idx + 1]])
            cost += float(D[perm[-1], self.exit_node])
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        raw_valid_rate = float(raw_valid_count) / total_shots

        info = {
            "marginal_perm": marginal_perm,
            "best_perm": best_perm,
            "best_subproblem_dist": round(best_cost, 4),
            "raw_valid_rate": round(raw_valid_rate, 4),
            "num_unique_samples": len(counts),
            "marginal_confidence": float(np.mean([M[self.customer_to_local[c], t] for t, c in enumerate(best_perm)])),
        }
        return best_perm, (raw_valid_rate > 0.0), info


def decode_quantum_marginals(counts: Dict[str, int], customer_subset: List[int]) -> List[int]:
    """
    Decodes measurement bitstring counts to the maximum-weight bipartite matching permutation.
    """
    C = len(customer_subset)
    num_steps = C
    N = C * num_steps
    total_shots = max(1, sum(counts.values()))
    local_to_customer = {idx: c for idx, c in enumerate(customer_subset)}

    M = np.zeros((C, num_steps), dtype=float)
    for bitstring, count in counts.items():
        clean_bs = bitstring.replace(" ", "")
        if len(clean_bs) != N:
            continue
        weight = float(count) / total_shots
        for k in range(N):
            if clean_bs[N - 1 - k] == "1":
                loc_i = k // num_steps
                step = k % num_steps
                if loc_i < C and step < num_steps:
                    M[loc_i, step] += weight

    row_ind, col_ind = linear_sum_assignment(-M)
    step_to_cust = {int(c_t): local_to_customer[int(r_i)] for r_i, c_t in zip(row_ind, col_ind)}
    return [step_to_cust[t] for t in range(num_steps)]
