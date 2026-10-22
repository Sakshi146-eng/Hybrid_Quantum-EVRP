"""
QAOA Solver for Small Routing Instances using Qiskit and Qiskit Aer.
Executes a parameterized Quantum Approximate Optimization Algorithm (QAOA)
on a classical simulator (AerSimulator) to optimize routing bitstrings.

Important:
This is a hybrid quantum-classical optimization method running on a classical simulator
(Qiskit Aer). It does NOT run on physical quantum hardware.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

from src.vrp.instance import VRPInstance
from src.classical.base_solver import BaseVRPSolver
from src.classical.solution import VRPSolution
from src.quantum.qubo import VRPQuboBuilder


class QAOAVRPSolver(BaseVRPSolver):
    """
    Simulated QAOA Solver for Small Vehicle Routing Instances.
    """

    def __init__(
        self,
        p_depth: int = 1,
        shots: int = 1024,
        max_iter: int = 50,
        seed: Optional[int] = 42,
        penalty_multiplier: float = 2.0,
        penalty_weight: Optional[float] = None,
        max_allowed_customers: int = 4,
        **kwargs,
    ):
        """
        Args:
            p_depth (int): QAOA circuit depth / number of alternating layers (p).
            shots (int): Number of measurement shots on AerSimulator.
            max_iter (int): Maximum classical optimizer iterations (COBYLA).
            seed (Optional[int]): Random seed for reproducibility.
            penalty_multiplier (float): Adaptive QUBO penalty multiplier.
            penalty_weight (Optional[float]): Explicit QUBO penalty weight.
            max_allowed_customers (int): Safety threshold on problem size (default 4 -> 16 qubits).
        """
        super().__init__(seed=seed, **kwargs)
        self.p_depth = max(1, int(p_depth))
        self.shots = max(100, int(shots))
        self.max_iter = max(10, int(max_iter))
        self.penalty_multiplier = penalty_multiplier
        self.penalty_weight = penalty_weight
        self.max_allowed_customers = max_allowed_customers

        self.last_metadata: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "QAOA Simulator"

    def _build_circuit(
        self,
        builder: VRPQuboBuilder,
        single_qubit_terms: List[Tuple[int, float]],
        two_qubit_terms: List[Tuple[int, int, float]],
        gammas: List[float],
        betas: List[float],
        add_measurements: bool = False,
    ) -> QuantumCircuit:
        """
        Constructs the parameterized QAOA quantum circuit.

        Args:
            builder: QUBO builder with variable counts and mappings.
            single_qubit_terms: List of (qubit_k, coefficient h_k).
            two_qubit_terms: List of (qubit_j, qubit_k, coefficient J_{jk}).
            gammas: Angle parameters for problem Hamiltonian (length p).
            betas: Angle parameters for mixer Hamiltonian (length p).
            add_measurements: Whether to append measurement operations to all qubits.

        Returns:
            QuantumCircuit: Constructed Qiskit circuit.
        """
        num_qubits = builder.num_vars
        qc = QuantumCircuit(num_qubits)

        # 1. Initial State: Equal superposition |+>^N via Hadamard gates
        qc.h(range(num_qubits))

        # 2. Alternating QAOA Layers
        for layer in range(self.p_depth):
            gamma = gammas[layer]
            beta = betas[layer]

            # a. Problem Unitary U(H_C, gamma) = exp(-i * gamma * H_C)
            # Single-qubit Z rotations: Rz(2 * gamma * h_k) on qubit k
            for k, h_k in single_qubit_terms:
                qc.rz(2.0 * gamma * h_k, k)

            # Two-qubit ZZ interactions: Rzz(2 * gamma * J_jk) on qubits j and k
            for j, k, J_jk in two_qubit_terms:
                qc.rzz(2.0 * gamma * J_jk, j, k)

            # b. Mixer Unitary U(H_M, beta) = exp(-i * beta * H_M), H_M = sum X_k
            # Rx(2 * beta) on each qubit
            for k in range(num_qubits):
                qc.rx(2.0 * beta, k)

        # 3. Optional Measurements
        if add_measurements:
            qc.measure_all()

        return qc

    def _solve_instance(self, instance: VRPInstance) -> List[List[int]]:
        """
        Executes QAOA algorithm pipeline for a full VRPInstance:
        1. Formulates QUBO and extracts Ising coefficients.
        2. Runs COBYLA classical parameter optimizer on simulator expectation value.
        3. Samples measurement bitstrings using Qiskit Aer.
        4. Decodes and validates candidate routes.
        """
        builder = VRPQuboBuilder(
            instance=instance,
            penalty_multiplier=self.penalty_multiplier,
            penalty_weight=self.penalty_weight,
        )
        chosen_route, _ = self.solve_builder(builder)
        return [chosen_route]

    def solve_builder(self, builder: VRPQuboBuilder) -> Tuple[List[int], Dict[str, Any]]:
        """
        Executes QAOA on an arbitrary VRPQuboBuilder (full instance or subproblem).
        Returns:
            Tuple[List[int], Dict[str, Any]]: (chosen_route, metadata)
        """
        num_cust = builder.num_customers
        instance = builder.instance
        if num_cust < 2:
            raise ValueError(f"QAOA routing requires >= 2 customers, got {num_cust}.")
        if num_cust > self.max_allowed_customers:
            raise ValueError(
                f"Instance has {num_cust} customers, requiring {num_cust**2} qubits. "
                f"Classical simulation limit is {self.max_allowed_customers} customers "
                f"({self.max_allowed_customers**2} qubits) to maintain interactive performance."
            )

        t_start_total = time.perf_counter()

        # Step 1: QUBO and Ising formulation
        t0_circuit = time.perf_counter()
        Q, offset_qubo = builder.build_qubo()
        hamiltonian, ising_offset = builder.to_ising_hamiltonian()

        # Extract explicit single and two-qubit terms for direct circuit building
        num_qubits = builder.num_vars
        single_qubit_terms: List[Tuple[int, float]] = []
        two_qubit_terms: List[Tuple[int, int, float]] = []

        # From QUBO upper-triangular:
        # H_C = sum h_k Z_k + sum J_{jk} Z_j Z_k
        # J_{jk} = Q[j, k] / 4
        # h_k = -Q[k, k] / 2 - sum_{j < k} Q[j, k] / 4 - sum_{k < l} Q[k, l] / 4
        for k in range(num_qubits):
            h_k = -0.5 * Q[k, k]
            for j in range(k):
                h_k -= 0.25 * Q[j, k]
            for l in range(k + 1, num_qubits):
                h_k -= 0.25 * Q[k, l]
            if abs(h_k) > 1e-12:
                single_qubit_terms.append((k, float(h_k)))

        for j in range(num_qubits):
            for k in range(j + 1, num_qubits):
                J_jk = 0.25 * Q[j, k]
                if abs(J_jk) > 1e-12:
                    two_qubit_terms.append((j, k, float(J_jk)))

        t_circuit_prep = time.perf_counter() - t0_circuit

        # Step 2: Hamiltonian scaling and Classical Parameter Optimization (COBYLA)
        # Normalize problem Hamiltonian terms by max coefficient so that gamma angle rotations
        # operate smoothly on the [0, pi] scale rather than oscillating violently.
        max_coeff = max(
            max([abs(h) for _, h in single_qubit_terms], default=1.0),
            max([abs(J) for _, _, J in two_qubit_terms], default=1.0),
        )
        scale = float(max_coeff) if max_coeff > 1e-12 else 1.0

        norm_single = [(k, h / scale) for k, h in single_qubit_terms]
        norm_two = [(j, k, J / scale) for j, k, J in two_qubit_terms]

        t0_opt = time.perf_counter()
        eval_count = 0

        def objective_function(params: np.ndarray) -> float:
            nonlocal eval_count
            eval_count += 1
            gammas = params[: self.p_depth]
            betas = params[self.p_depth :]

            qc = self._build_circuit(
                builder=builder,
                single_qubit_terms=norm_single,
                two_qubit_terms=norm_two,
                gammas=gammas,
                betas=betas,
                add_measurements=False,
            )
            # Fast expectation value evaluation using Statevector
            sv = Statevector.from_instruction(qc)
            exp_energy = float(sv.expectation_value(hamiltonian).real) + ising_offset
            return exp_energy

        # Multi-start initial parameter optimization to discover good convergence basins
        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState(42)
        best_opt = None
        best_energy = float("inf")

        num_starts = 2 if self.max_iter >= 25 else 1
        per_start_iters = max(10, self.max_iter // num_starts)

        for _ in range(num_starts):
            init_params = np.concatenate([
                rng.uniform(0.1, np.pi, self.p_depth),
                rng.uniform(0.1, np.pi / 2.0, self.p_depth),
            ])
            res = minimize(
                objective_function,
                init_params,
                method="COBYLA",
                options={"maxiter": per_start_iters, "tol": 1e-3},
            )
            if res.fun < best_energy:
                best_energy = float(res.fun)
                best_opt = res

        optimal_params = best_opt.x
        opt_gammas = list(optimal_params[: self.p_depth])
        opt_betas = list(optimal_params[self.p_depth :])
        t_opt = time.perf_counter() - t0_opt

        # Step 3: Circuit Sampling on AerSimulator
        t0_sample = time.perf_counter()
        measured_qc = self._build_circuit(
            builder=builder,
            single_qubit_terms=norm_single,
            two_qubit_terms=norm_two,
            gammas=opt_gammas,
            betas=opt_betas,
            add_measurements=True,
        )

        sim = AerSimulator()
        job = sim.run(measured_qc, shots=self.shots, seed_simulator=self.seed)
        result = job.result()
        counts: Dict[str, int] = result.get_counts()
        t_sample = time.perf_counter() - t0_sample

        # Step 4: Decode, Validate, and Account for Raw vs Repaired Candidates
        total_samples = sum(counts.values())
        valid_samples = 0
        best_valid_route: Optional[List[int]] = None
        best_valid_distance = float("inf")
        best_valid_bitstring = None

        candidate_samples: List[Dict[str, Any]] = []

        # Sort bitstrings by frequency
        sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

        for bitstring, count in sorted_counts:
            route, is_valid, details = builder.decode_bitstring(bitstring)
            energy = builder.evaluate_bitstring_energy(bitstring)

            if is_valid and route is not None:
                valid_samples += count
                dist = instance.calculate_route_distance(route)
                if dist < best_valid_distance:
                    best_valid_distance = dist
                    best_valid_route = route
                    best_valid_bitstring = bitstring
            else:
                dist = None

            if len(candidate_samples) < 10:
                rep_route, was_rep, _ = builder.repair_bitstring_to_route(bitstring)
                rep_dist = instance.calculate_route_distance(rep_route)
                candidate_samples.append({
                    "bitstring": bitstring,
                    "count": count,
                    "probability": round(count / total_samples, 4),
                    "is_valid": is_valid,
                    "raw_distance": round(dist, 4) if dist is not None else None,
                    "repaired_distance": round(rep_dist, 4),
                    "energy": round(energy, 4),
                    "route": route if is_valid else rep_route,
                })

        raw_valid_sample_rate = float(valid_samples / total_samples) if total_samples > 0 else 0.0

        repair_applied = False
        fallback_used = False
        best_raw_distance = None
        best_repaired_distance = None

        if best_valid_route is not None:
            chosen_route = best_valid_route
            chosen_is_feasible = True
            best_raw_distance = float(best_valid_distance)
        else:
            # If no raw valid sample was sampled, do NOT claim raw QAOA feasibility.
            # Perform explicit documented repair on top sample for visualization fallback.
            chosen_is_feasible = False
            top_bs = sorted_counts[0][0]
            repaired_route, was_rep, rep_details = builder.repair_bitstring_to_route(top_bs)
            chosen_route = repaired_route
            repair_applied = was_rep
            fallback_used = True
            best_repaired_distance = float(instance.calculate_route_distance(repaired_route))

        t_total = time.perf_counter() - t_start_total

        self.last_metadata = {
            "p_depth": self.p_depth,
            "qubits": num_qubits,
            "shots": self.shots,
            "valid_samples": valid_samples,
            "total_samples": total_samples,
            "valid_sample_rate": round(raw_valid_sample_rate * 100.0, 2),
            "raw_valid_sample_rate": round(raw_valid_sample_rate * 100.0, 2),
            "repaired_solution_rate": round(100.0 if fallback_used else 0.0, 2),
            "best_raw_sample_distance": round(best_raw_distance, 4) if best_raw_distance is not None else None,
            "best_repaired_distance": round(best_repaired_distance, 4) if best_repaired_distance is not None else None,
            "repair_applied": repair_applied,
            "fallback_used": fallback_used,
            "optimizer_evaluations": eval_count,
            "optimal_gammas": [round(float(g), 4) for g in opt_gammas],
            "optimal_betas": [round(float(b), 4) for b in opt_betas],
            "min_energy_found": round(best_energy, 4),
            "penalty_weight": round(builder.penalty_weight, 4),
            "hamiltonian_scale": round(scale, 4),
            "timing_breakdown": {
                "circuit_prep_seconds": round(t_circuit_prep, 6),
                "optimization_seconds": round(t_opt, 6),
                "sampling_seconds": round(t_sample, 6),
                "total_seconds": round(t_total, 6),
            },
            "top_candidates": candidate_samples,
            "best_valid_bitstring": best_valid_bitstring,
            "counts": counts,
            "device": "Qiskit Aer Simulator (Classical Emulation)",
        }
        self.last_is_feasible = chosen_is_feasible

        return chosen_route, self.last_metadata

    def solve(self, instance: VRPInstance) -> VRPSolution:
        """Solves instance with QAOA and attaches quantum metadata."""
        solution = super().solve(instance)
        # If raw QAOA sampled zero valid candidates, ensure feasibility is accurately marked False
        if hasattr(self, "last_is_feasible") and not self.last_is_feasible:
            solution.feasible = False
        if hasattr(self, "last_metadata"):
            solution.metadata.update(self.last_metadata)
        return solution
