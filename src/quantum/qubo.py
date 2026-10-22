"""
QUBO Formulation for Small Routing Instances.
Formulates a single-vehicle VRP / TSP tour over a depot and selected customer nodes
into a Quadratic Unconstrained Binary Optimization (QUBO) matrix and corresponding Ising Hamiltonian.

Encoding:
- Position/step-based encoding: C customers and C time steps.
- Binary variable x_{i, t} in {0, 1} indicates whether customer i is visited at step t in the route.
- Total binary variables (qubits) N = C^2.
- Single vehicle route starts at depot (node 0) at step -1, visits step 0..C-1, returns to depot at step C.
"""
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from src.vrp.instance import VRPInstance


def extract_sub_instance(
    parent_instance: VRPInstance,
    customer_indices: Optional[List[int]] = None,
    customer_count: Optional[int] = None,
) -> VRPInstance:
    """
    Extracts a small, self-contained VRPInstance containing the depot (node 0)
    and a designated subset of customer nodes from a parent benchmark instance.

    Args:
        parent_instance (VRPInstance): Full benchmark instance (e.g. from CEC-2020).
        customer_indices (Optional[List[int]]): Explicit customer indices from parent (e.g. [1, 2, 3]).
        customer_count (Optional[int]): Number of customers to slice (e.g. first 3 customers) if indices not given.

    Returns:
        VRPInstance: A new, independent VRPInstance with mapped node indices [0, 1, ..., C].
    """
    if customer_indices is None:
        if customer_count is None:
            customer_count = min(3, parent_instance.num_customers)
        # Exclude depot (0)
        available = [i for i in range(parent_instance.num_nodes) if i != parent_instance.depot_idx]
        customer_indices = available[:customer_count]

    # Validate customer indices
    for cid in customer_indices:
        if cid == parent_instance.depot_idx:
            raise ValueError(f"Depot index {cid} cannot be in customer_indices.")
        if cid < 0 or cid >= parent_instance.num_nodes:
            raise ValueError(f"Customer index {cid} out of bounds for parent instance.")

    # Sub-instance node order: depot (0) followed by selected customers
    sub_nodes = [parent_instance.depot_idx] + list(customer_indices)
    num_sub_nodes = len(sub_nodes)

    sub_coords = (
        parent_instance.coordinates[sub_nodes].copy()
        if parent_instance.coordinates is not None
        else None
    )
    sub_dist_matrix = parent_instance.distance_matrix[np.ix_(sub_nodes, sub_nodes)].copy()
    sub_demands = parent_instance.demands[sub_nodes].copy()
    sub_demands[0] = 0.0

    # Ensure single vehicle capacity is sufficient for sub-instance comparison
    tot_demand = float(sub_demands.sum())
    cap = parent_instance.vehicle_capacity
    if cap is None or cap < tot_demand:
        cap = tot_demand + 10.0

    sub_instance = VRPInstance(
        distance_matrix=sub_dist_matrix,
        coordinates=sub_coords,
        num_vehicles=1,
        depot_idx=0,
        demands=sub_demands,
        vehicle_capacity=cap,
        distance_metric=parent_instance.distance_metric,
    )

    parent_name = (
        parent_instance.raw_data.get("name", "instance")
        if hasattr(parent_instance, "raw_data") and isinstance(parent_instance.raw_data, dict)
        else "instance"
    )
    sub_instance.raw_data = {
        "name": f"{parent_name}_sub_{len(customer_indices)}c",
        "parent_instance": parent_name,
        "selected_customers_parent_ids": customer_indices,
    }

    return sub_instance


class VRPQuboBuilder:
    """
    Constructs the QUBO formulation and Ising Hamiltonian for a routing problem or subproblem.
    Supports both full tours (depot -> customers -> depot) and local route subproblems
    with defined entry and exit nodes (entry_node -> customer_subset -> exit_node).
    """

    def __init__(
        self,
        instance: VRPInstance,
        penalty_multiplier: float = 2.0,
        penalty_weight: Optional[float] = None,
        entry_node: Optional[int] = None,
        exit_node: Optional[int] = None,
        customer_subset: Optional[List[int]] = None,
    ):
        """
        Args:
            instance (VRPInstance): Problem instance containing distance matrix and node data.
            penalty_multiplier (float): Multiplier on max distance for adaptive penalty weight (default 2.0).
            penalty_weight (Optional[float]): Explicit penalty weight override.
            entry_node (Optional[int]): Origin node preceding the sequence (defaults to instance.depot_idx).
            exit_node (Optional[int]): Destination node following the sequence (defaults to instance.depot_idx).
            customer_subset (Optional[List[int]]): Specific customer subset to optimize (defaults to all customers).
        """
        self.instance = instance
        self.depot_idx = instance.depot_idx
        self.entry_node = entry_node if entry_node is not None else self.depot_idx
        self.exit_node = exit_node if exit_node is not None else self.depot_idx

        if customer_subset is not None:
            self.customer_indices = list(customer_subset)
        else:
            self.customer_indices = [i for i in range(instance.num_nodes) if i != self.depot_idx]

        self.num_customers = len(self.customer_indices)
        self.num_steps = self.num_customers

        if self.num_customers < 2:
            raise ValueError(f"QUBO routing requires at least 2 customers, got {self.num_customers}.")

        # Total binary variables N = C * C
        self.num_vars = self.num_customers * self.num_steps

        # Map customer index (in instance) and step (0..C-1) to variable index k in [0..N-1]
        self.customer_to_local: Dict[int, int] = {
            cust_id: idx for idx, cust_id in enumerate(self.customer_indices)
        }
        self.local_to_customer: Dict[int, int] = {
            idx: cust_id for idx, cust_id in enumerate(self.customer_indices)
        }

        # Calculate calibrated penalty weight:
        # Omitting a customer saves at most 2 * max_dist by triangle inequality.
        # Penalty P = penalty_multiplier * max_dist with multiplier >= 2.0 guarantees that any valid permutation
        # has strictly lower energy than any invalid assignment, without inflating the Hamiltonian norm.
        max_dist = float(np.max(instance.distance_matrix))
        if penalty_weight is not None and penalty_weight > 0:
            self.penalty_weight = float(penalty_weight)
        else:
            self.penalty_weight = float(max_dist * max(2.0, penalty_multiplier))

        self._Q: Optional[np.ndarray] = None
        self._offset: float = 0.0

    def get_var_index(self, customer_node: int, step: int) -> int:
        """
        Returns flat variable index for (customer_node, step).
        Variable index k = local_customer_idx * C + step
        """
        if customer_node not in self.customer_to_local:
            raise KeyError(f"Customer {customer_node} not in customer set {self.customer_indices}")
        if not (0 <= step < self.num_steps):
            raise IndexError(f"Step {step} out of bounds [0, {self.num_steps - 1}]")
        loc_i = self.customer_to_local[customer_node]
        return loc_i * self.num_steps + step

    def get_customer_and_step(self, var_index: int) -> Tuple[int, int]:
        """Inverse mapping from flat variable index to (customer_node, step)."""
        if not (0 <= var_index < self.num_vars):
            raise IndexError(f"Variable index {var_index} out of bounds [0, {self.num_vars - 1}]")
        loc_i = var_index // self.num_steps
        step = var_index % self.num_steps
        cust_node = self.local_to_customer[loc_i]
        return cust_node, step

    def build_qubo(self) -> Tuple[np.ndarray, float]:
        """
        Builds upper-triangular QUBO matrix Q such that Objective = x^T Q x + offset.

        Returns:
            Tuple[np.ndarray, float]:
                - Q: Upper-triangular numpy array of shape (num_vars, num_vars).
                - offset: Constant energy offset.
        """
        C = self.num_customers
        N = self.num_vars
        P = self.penalty_weight
        D = self.instance.distance_matrix
        entry = self.entry_node
        exit_pt = self.exit_node

        # Initialize upper-triangular matrix
        Q = np.zeros((N, N), dtype=float)
        offset = 0.0

        # Helper to add term to Q in upper-triangular convention
        def add_term(u: int, v: int, val: float):
            if u == v:
                Q[u, u] += val
            else:
                lo, hi = (u, v) if u < v else (v, u)
                Q[lo, hi] += val

        # -----------------------------------------------------------------
        # 1. Routing Objective Terms: Minimize Total Distance
        # -----------------------------------------------------------------
        # a. From entry_node to step 0
        for i in self.customer_indices:
            k = self.get_var_index(i, 0)
            add_term(k, k, float(D[entry, i]))

        # b. Between consecutive steps t and t + 1
        for t in range(self.num_steps - 1):
            for i in self.customer_indices:
                k_i = self.get_var_index(i, t)
                for j in self.customer_indices:
                    if i != j:
                        k_j = self.get_var_index(j, t + 1)
                        add_term(k_i, k_j, float(D[i, j]))

        # c. From last step (C-1) to exit_node
        last_step = self.num_steps - 1
        for i in self.customer_indices:
            k = self.get_var_index(i, last_step)
            add_term(k, k, float(D[i, exit_pt]))

        # -----------------------------------------------------------------
        # 2. Penalty Constraint 1: Each customer visited exactly once
        #    P * sum_i ( sum_t x_{i, t} - 1 )^2
        #    = P * sum_i [ 1 - sum_t x_{i, t} + 2 * sum_{t1 < t2} x_{i, t1} x_{i, t2} ]
        # -----------------------------------------------------------------
        for i in self.customer_indices:
            # Constant offset +P per customer
            offset += P
            # Linear terms: -P * x_{i, t}
            for t in range(self.num_steps):
                k = self.get_var_index(i, t)
                add_term(k, k, -P)
            # Quadratic cross-terms: +2P * x_{i, t1} * x_{i, t2}
            for t1 in range(self.num_steps):
                for t2 in range(t1 + 1, self.num_steps):
                    k1 = self.get_var_index(i, t1)
                    k2 = self.get_var_index(i, t2)
                    add_term(k1, k2, 2.0 * P)

        # -----------------------------------------------------------------
        # 3. Penalty Constraint 2: Each step has exactly one customer
        #    P * sum_t ( sum_i x_{i, t} - 1 )^2
        #    = P * sum_t [ 1 - sum_i x_{i, t} + 2 * sum_{i1 < i2} x_{i1, t} x_{i2, t} ]
        # -----------------------------------------------------------------
        for t in range(self.num_steps):
            # Constant offset +P per time step
            offset += P
            # Linear terms: -P * x_{i, t}
            for i in self.customer_indices:
                k = self.get_var_index(i, t)
                add_term(k, k, -P)
            # Quadratic cross-terms: +2P * x_{i1, t} * x_{i2, t}
            for idx1 in range(len(self.customer_indices)):
                for idx2 in range(idx1 + 1, len(self.customer_indices)):
                    c1 = self.customer_indices[idx1]
                    c2 = self.customer_indices[idx2]
                    k1 = self.get_var_index(c1, t)
                    k2 = self.get_var_index(c2, t)
                    add_term(k1, k2, 2.0 * P)

        self._Q = Q
        self._offset = offset
        return Q, offset

    def to_ising_hamiltonian(self) -> Tuple[SparsePauliOp, float]:
        """
        Converts the QUBO problem into an Ising Hamiltonian:
        x_k = (I - Z_k) / 2

        Returns:
            Tuple[SparsePauliOp, float]:
                - H_C: Qiskit SparsePauliOp representing the cost Hamiltonian.
                - ising_offset: Constant energy shift.
        """
        if self._Q is None:
            self.build_qubo()

        Q = self._Q
        offset = self._offset
        N = self.num_vars

        # Coefficients for Pauli operators
        pauli_terms: List[Tuple[str, float]] = []
        ising_offset = float(offset)

        # 1. Linear terms Q[k, k] * (I - Z_k) / 2
        for k in range(N):
            val = float(Q[k, k])
            if abs(val) > 1e-12:
                ising_offset += 0.5 * val
                # Pauli Z on qubit k: string of 'I' with 'Z' at index (N - 1 - k)
                label = ["I"] * N
                label[N - 1 - k] = "Z"
                pauli_terms.append(("".join(label), -0.5 * val))

        # 2. Quadratic terms Q[j, k] * (I - Z_j)(I - Z_k) / 4
        for j in range(N):
            for k in range(j + 1, N):
                val = float(Q[j, k])
                if abs(val) > 1e-12:
                    # Offset contribution
                    ising_offset += 0.25 * val
                    # -0.25 * val * Z_j
                    label_j = ["I"] * N
                    label_j[N - 1 - j] = "Z"
                    pauli_terms.append(("".join(label_j), -0.25 * val))
                    # -0.25 * val * Z_k
                    label_k = ["I"] * N
                    label_k[N - 1 - k] = "Z"
                    pauli_terms.append(("".join(label_k), -0.25 * val))
                    # +0.25 * val * Z_j Z_k
                    label_jk = ["I"] * N
                    label_jk[N - 1 - j] = "Z"
                    label_jk[N - 1 - k] = "Z"
                    pauli_terms.append(("".join(label_jk), 0.25 * val))

        # Consolidate Pauli terms
        if pauli_terms:
            hamiltonian = SparsePauliOp.from_list(pauli_terms).simplify()
        else:
            hamiltonian = SparsePauliOp.from_list([("I" * N, 0.0)])

        return hamiltonian, ising_offset

    def bitstring_to_vector(self, bitstring: str) -> np.ndarray:
        """
        Converts Qiskit measurement bitstring (e.g. '010010001') to binary variable vector x.
        Note: Qiskit ordering is right-to-left (bitstring[-1] is qubit 0).
        """
        N = self.num_vars
        clean_bs = bitstring.replace(" ", "")
        if len(clean_bs) != N:
            raise ValueError(f"Bitstring length {len(clean_bs)} does not match num_vars {N}")

        x = np.zeros(N, dtype=float)
        for k in range(N):
            # Qubit k corresponds to index N - 1 - k in Qiskit bitstring
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
    ) -> Tuple[List[int], bool, Dict[str, Any]]:
        """
        Decodes a binary measurement bitstring into a vehicle route.

        Args:
            bitstring (str): Bitstring result from QAOA measurement.

        Returns:
            Tuple[List[int], bool, Dict[str, Any]]:
                - route: Ordered node indices starting and ending at depot [0, ..., 0].
                - is_valid: True if bitstring satisfies exact one-hot permutation constraints.
                - details: Diagnostic details (active variables, collisions, unassigned steps).
        """
        if self._Q is None:
            self.build_qubo()

        x = self.bitstring_to_vector(bitstring)
        C = self.num_customers
        depot = self.depot_idx

        # Group assignments by step
        step_assignments: Dict[int, List[int]] = {t: [] for t in range(self.num_steps)}
        customer_visits: Dict[int, List[int]] = {c: [] for c in self.customer_indices}

        for k, val in enumerate(x):
            if val > 0.5:
                cust, step = self.get_customer_and_step(k)
                step_assignments[step].append(cust)
                customer_visits[cust].append(step)

        # Check validity
        is_valid = True
        violations = []

        # Check each step has exactly one customer
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

        # Check each customer visited exactly once
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
        """
        Explicitly repairs an invalid candidate bitstring into a valid permutation route.
        Documents all repair operations taken without disguising them as raw samples.

        Args:
            bitstring (str): Raw candidate bitstring.

        Returns:
            Tuple[List[int], bool, Dict[str, Any]]:
                - repaired_route: Valid route starting at entry_node and ending at exit_node.
                - repair_applied: True if repair was required; False if already valid.
                - repair_metadata: Detailed log of conflicts and missing customers filled.
        """
        raw_route, is_valid, details = self.decode_bitstring(bitstring)
        if is_valid and raw_route is not None:
            return raw_route, False, {"repair_applied": False, "actions": []}

        step_assignments = details["step_assignments"]
        used = set()
        route_customers = []
        actions = []

        for t in range(self.num_steps):
            custs = step_assignments[t]
            chosen = None
            for c in custs:
                if c not in used:
                    chosen = c
                    break
            if chosen is not None:
                route_customers.append(chosen)
                used.add(chosen)
            else:
                actions.append(f"Step {t} had no unassigned candidate.")

        # Fill remaining missing customers
        missing = [c for c in self.customer_indices if c not in used]
        if missing:
            actions.append(f"Inserted missing customers {missing}.")
            route_customers.extend(missing)

        repaired_route = [self.entry_node] + route_customers + [self.exit_node]
        return repaired_route, True, {
            "repair_applied": True,
            "missing_customers": missing,
            "actions": actions,
            "original_violations": details["violations"],
        }
