"""
qaoa_qiskit.py  -  QAOA implemented entirely with Qiskit.

  circuit      : |+>^n, then p x [ cost layer RZ/RZZ(gamma) , mixer RX(2 beta) ]
  objective    : <H_QUBO> from Aer's save_expectation_value on the parametrised circuit
                 (exact statevector simulation, circuit transpiled once, parameters re-bound each step)
  optimiser    : SciPy COBYLA (classical outer loop; this is what the RL agent replaces later)
  measurement  : Aer simulator, `shots` shots (like the paper's 1024/2048), plus exact
                 probabilities via Aer's save_probabilities for the "chance of the optimum" metric
"""
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from vrp_qubo import route_cost


def build_qaoa_circuit(qubo, p):
    """Parametrised QAOA circuit for the QUBO's Ising Hamiltonian."""
    const, h, J = qubo.to_ising()
    n = qubo.n
    gam, bet = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for l in range(p):
        for i, hi in enumerate(h):
            if hi != 0.0:
                qc.rz(2 * gam[l] * float(hi), i)              # exp(-i gamma h Z)
        for (i, j), Jij in J.items():
            if Jij != 0.0:
                qc.rzz(2 * gam[l] * float(Jij), i, j)         # exp(-i gamma J Z Z)
        qc.rx(2 * bet[l], range(n))                           # mixer exp(-i beta X)
    return qc


class QAOA:
    def __init__(self, enc, p, seed=0):
        self.enc, self.p, self.n = enc, p, enc.N
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.circuit = build_qaoa_circuit(enc.qubo, p)
        self.observable, self.const = enc.qubo.ising_operator()
        self.backend = AerSimulator(method="statevector")
        qc = self.circuit.copy()
        qc.save_expectation_value(self.observable, list(range(self.n)))
        self.exp_circuit = transpile(qc, self.backend, optimization_level=0)

    # -- objective ------------------------------------------------------------------
    def energy(self, params):
        """<H_QUBO> for the given (gamma_1..gamma_p, beta_1..beta_p)."""
        bound = self.exp_circuit.assign_parameters(np.asarray(params, float))
        ev = self.backend.run(bound).result().data()["expectation_value"]
        return float(ev) + self.const

    # -- classical outer loop ---------------------------------------------------------
    def optimise(self, restarts=3, maxiter=120):
        from scipy.optimize import minimize
        best = None
        for r in range(restarts):
            x0 = np.concatenate([self.rng.uniform(0.05, 1.0, self.p), self.rng.uniform(0, np.pi / 2, self.p)])
            trace = []

            def f(x):
                v = self.energy(x)
                trace.append(v)
                return v

            res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
            if best is None or res.fun < best["fun"]:
                best = dict(x=res.x, fun=float(res.fun), trace=trace)
        self.best = best
        return best

    # -- measurement ------------------------------------------------------------------
    def bound(self, params):
        return self.circuit.assign_parameters(np.asarray(params, float))

    def probabilities(self, params):
        """Exact outcome probabilities of the final state (Aer save_probabilities)."""
        qc = self.bound(params)
        qc.save_probabilities()
        res = self.backend.run(transpile(qc, self.backend, optimization_level=0)).result()
        return np.asarray(res.data()["probabilities"])

    def sample(self, params, shots=2048):
        """Measure `shots` times. Returns {bitstring_int: count}."""
        qc = self.bound(params)
        qc.measure_all()
        res = self.backend.run(transpile(qc, self.backend, optimization_level=0),
                               shots=shots, seed_simulator=self.seed).result()
        return {int(k.replace(" ", ""), 2): v for k, v in res.get_counts().items()}


def evaluate_counts(enc, counts, D, opt_cost):
    """Decode measured bitstrings; report feasibility, hit rate of the optimum, best cost."""
    tot = sum(counts.values())
    feas = hit = 0
    best = np.inf
    for b, c in counts.items():
        r = enc.decode(b)
        if r is None:
            continue
        cost = route_cost(D, r)
        feas += c
        best = min(best, cost)
        if cost <= opt_cost + 1e-6:
            hit += c
    return dict(feasible_frac=feas / tot, optimal_frac=hit / tot,
                best_cost=None if best == np.inf else float(best))


def run_qaoa(enc, D, opt_cost, optimal_routes, p, shots=2048, restarts=3, maxiter=120, seed=0):
    t0 = time.time()
    q = QAOA(enc, p, seed)
    best = q.optimise(restarts, maxiter)
    probs = q.probabilities(best["x"])
    ground = [enc.encode(r) for r in optimal_routes]
    p_opt = float(probs[ground].sum())
    uniform = len(ground) / (1 << enc.N)
    e_min, e_mean = opt_cost / enc.scale, enc.qubo.mean_energy()
    ev = evaluate_counts(enc, q.sample(best["x"], shots), D, opt_cost)
    gap = None if ev["best_cost"] is None else round(max(0.0, 100 * (ev["best_cost"] - opt_cost) / opt_cost), 4)
    return dict(p=p, params=best["x"].tolist(), expected_energy=best["fun"],
                energy_improvement=(e_mean - best["fun"]) / (e_mean - e_min),   # 0 = random, 1 = perfect
                prob_optimal=p_opt, boost_vs_uniform=p_opt / uniform,
                evaluations=len(best["trace"]), gap_pct=gap, seconds=time.time() - t0, **ev), best["trace"]
