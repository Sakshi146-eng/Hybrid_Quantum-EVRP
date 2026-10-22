"""
Test QAOA on 4 customers with Hamiltonian normalization / proper gamma scaling.
"""
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e_cvrp_loader import load_e_cvrp_instance
from src.quantum.qubo import extract_sub_instance, VRPQuboBuilder

def test_normalized_qaoa(p_depth=2, shots=4096, seed=42):
    inst = load_e_cvrp_instance("data/raw/e_cvrp/X-n360-k40-s9.evrp")
    sub_inst = extract_sub_instance(inst, customer_count=4)
    max_d = float(np.max(sub_inst.distance_matrix))
    
    # Moderate penalty: P = 2.0 * max_d
    P = 2.0 * max_d
    builder = VRPQuboBuilder(sub_inst, penalty_weight=P)
    Q, offset = builder.build_qubo()
    H, ising_offset = builder.to_ising_hamiltonian()
    num_qubits = builder.num_vars
    
    # Extract h_k and J_jk
    single_qubit_terms = []
    two_qubit_terms = []
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
                
    max_coeff = max(
        max(abs(h) for _, h in single_qubit_terms),
        max(abs(J) for _, _, J in two_qubit_terms)
    )
    print(f"Max coeff in H: {max_coeff:.2f}")
    
    # Normalize by max_coeff
    scale = max_coeff
    norm_single = [(k, h / scale) for k, h in single_qubit_terms]
    norm_two = [(j, k, J / scale) for j, k, J in two_qubit_terms]
    
    def build_qc(gammas, betas, measure=False):
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))
        for layer in range(p_depth):
            g = gammas[layer]
            b = betas[layer]
            for k, h in norm_single:
                qc.rz(2.0 * g * h, k)
            for j, k, J in norm_two:
                qc.rzz(2.0 * g * J, j, k)
            for k in range(num_qubits):
                qc.rx(2.0 * b, k)
        if measure:
            qc.measure_all()
        return qc

    # Objective function
    def cost_fn(params):
        gammas = params[:p_depth]
        betas = params[p_depth:]
        qc = build_qc(gammas, betas, measure=False)
        sv = Statevector.from_instruction(qc)
        exp_val = float(sv.expectation_value(H).real) + ising_offset
        return exp_val

    # Multi-start optimization
    best_opt = None
    best_cost = float("inf")
    rng = np.random.RandomState(seed)
    
    for start in range(3):
        # Initial parameters in normalized range [0, pi] and [0, pi/2]
        init_p = np.concatenate([
            rng.uniform(0.1, np.pi, p_depth),
            rng.uniform(0.1, np.pi / 2, p_depth),
        ])
        res = minimize(cost_fn, init_p, method="COBYLA", options={"maxiter": 60})
        if res.fun < best_cost:
            best_cost = res.fun
            best_opt = res

    opt_gammas = best_opt.x[:p_depth]
    opt_betas = best_opt.x[p_depth:]
    print(f"Best optimized energy: {best_cost:.4f}")
    print(f"Optimal gammas: {opt_gammas}")
    print(f"Optimal betas: {opt_betas}")

    # Sample with AerSimulator
    qc_meas = build_qc(opt_gammas, opt_betas, measure=True)
    sim = AerSimulator()
    job = sim.run(qc_meas, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()

    valid_samples = 0
    valid_routes = {}
    total = sum(counts.values())

    for bs, cnt in counts.items():
        route, is_valid, details = builder.decode_bitstring(bs)
        if is_valid:
            valid_samples += cnt
            dist = sub_inst.calculate_route_distance(route)
            r_tuple = tuple(route)
            if r_tuple not in valid_routes or dist < valid_routes[r_tuple]["dist"]:
                valid_routes[r_tuple] = {"count": valid_routes.get(r_tuple, {}).get("count", 0) + cnt, "dist": dist, "bs": bs}

    print(f"\n--- Sampling Results ---")
    print(f"Total shots: {total}")
    print(f"Valid samples: {valid_samples} ({valid_samples / total * 100:.2f}%)")
    print(f"Distinct valid routes found: {len(valid_routes)}")
    for r, info in sorted(valid_routes.items(), key=lambda kv: kv[1]["dist"])[:5]:
        print(f"  Route: {list(r)}, Dist: {info['dist']:.4f}, Count: {info['count']}, Bitstring: {info['bs']}")

if __name__ == "__main__":
    test_normalized_qaoa(p_depth=2, shots=4096)
