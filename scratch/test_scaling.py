"""
Diagnostic: Test Hamiltonian scaling, penalty weights, and optimizer on 4 customers.
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

def test_scaling_and_penalties():
    inst = load_e_cvrp_instance("data/raw/e_cvrp/X-n360-k40-s9.evrp")
    sub_inst = extract_sub_instance(inst, customer_count=4)
    max_d = float(np.max(sub_inst.distance_matrix))
    print(f"Max dist: {max_d:.2f}")

    # Try different penalty weights: P = 1.2 * 2 * max_d vs 2 * 4 * max_d
    for P_factor in [2.5, 4.0, 8.0]:
        P = P_factor * max_d
        builder = VRPQuboBuilder(sub_inst, penalty_weight=P)
        Q, offset = builder.build_qubo()
        H, ising_offset = builder.to_ising_hamiltonian()
        
        # Check energy spectrum of valid vs invalid
        # Generate all 24 valid permutations
        import itertools
        valid_energies = []
        for perm in itertools.permutations([1, 2, 3, 4]):
            x = np.zeros(16)
            for p, c in enumerate(perm):
                x[builder.get_var_index(c, p)] = 1.0
            e = float(x.T @ Q @ x + offset)
            valid_energies.append(e)
            
        print(f"\n--- Penalty P={P:.1f} (factor={P_factor}) ---")
        print(f"Valid energy range: [{min(valid_energies):.2f}, {max(valid_energies):.2f}]")
        
        # Check invalid assignments (e.g. one omitted, one collision)
        x_invalid = np.zeros(16)
        x_invalid[builder.get_var_index(1, 0)] = 1.0
        x_invalid[builder.get_var_index(2, 0)] = 1.0 # collision
        x_invalid[builder.get_var_index(3, 1)] = 1.0
        e_inv = float(x_invalid.T @ Q @ x_invalid + offset)
        print(f"Sample invalid energy: {e_inv:.2f} (gap over max valid: {e_inv - max(valid_energies):.2f})")
        
        # What is the max Pauli coefficient in H?
        pauli_coeffs = [abs(c) for c in H.coeffs]
        max_pauli = max(pauli_coeffs)
        print(f"Max Pauli Z coeff in H: {max_pauli:.2f}")
        # Typical period for gamma
        gamma_period = np.pi / (2.0 * max_pauli)
        print(f"Estimated gamma period: {gamma_period:.6f} rad")

if __name__ == "__main__":
    test_scaling_and_penalties()
