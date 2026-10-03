"""
vrp_qubo.py  -  instances, QUBO encodings (arc / order) and QUBO -> Ising conversion.

Pure problem-formulation code (no quantum simulation here). The quantum part is in qaoa_qiskit.py.

Two encodings of a single-vehicle routing problem (depot = node 0):
  arc   : x_ij = 1  if the truck drives i -> j          (the paper's style; n*(n-1) qubits)
  order : x_ct = 1  if customer c is the t-th stop      ((n-1)^2 qubits)
"""
import itertools

import numpy as np
from qiskit.quantum_info import SparsePauliOp


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def load_instance(path):
    """Parse TSPLIB-like text files (Mavrovouniotis .evrp, CMT-style .vrp).
    VRP-REP XML files are not supported by this parser."""
    coords, demand, stations, depots, meta = {}, {}, set(), [], {}
    section = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue
            if line.upper().endswith("SECTION"):
                section = line.upper()
                continue
            if section is None:
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
                continue
            parts = line.split()
            if section == "NODE_COORD_SECTION":
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "DEMAND_SECTION":
                demand[int(parts[0])] = float(parts[1])
            elif section in ("STATIONS_COORD_SECTION", "STATION_COORD_SECTION"):
                stations.add(int(parts[0]))
            elif section == "DEPOT_SECTION":
                if int(parts[0]) != -1:
                    depots.append(int(parts[0]))
    depot = depots[0] if depots else min(coords)
    customers = sorted(i for i in coords if i != depot and i not in stations)
    return dict(depot=depot, customers=customers, coords=coords, demand=demand,
                stations=sorted(stations), meta=meta)


def dist_matrix(xy):
    xy = np.asarray(xy, float)
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))


def subsample(inst, k, mode="nearest", seed=0):
    """Depot + k customers. mode: 'nearest' (k closest to depot) | 'first' | 'random'."""
    dxy = np.array(inst["coords"][inst["depot"]])
    cust = inst["customers"]
    if mode == "first":
        chosen = cust[:k]
    elif mode == "random":
        chosen = [int(c) for c in np.random.default_rng(seed).choice(cust, size=k, replace=False)]
    else:
        d = [np.linalg.norm(np.array(inst["coords"][c]) - dxy) for c in cust]
        chosen = [cust[i] for i in np.argsort(d)[:k]]
    xy = [inst["coords"][inst["depot"]]] + [inst["coords"][c] for c in chosen]
    return np.array(xy), [inst["depot"]] + chosen


def random_instance(k, seed=0, box=100.0):
    xy = np.random.default_rng(seed).uniform(0, box, size=(k + 1, 2))
    return xy, list(range(k + 1))


def route_cost(D, route):
    return float(sum(D[a, b] for a, b in zip(route[:-1], route[1:])))


# ---------------------------------------------------------------------------
# QUBO
# ---------------------------------------------------------------------------
class QUBO:
    """E(b) = const + sum_i lin[i] b_i + sum_{i<j} quad[i,j] b_i b_j   with b in {0,1}."""

    def __init__(self, n):
        self.n, self.const = n, 0.0
        self.lin = np.zeros(n)
        self.quad = {}

    def add_quad(self, i, j, v):
        if i == j:                                   # b^2 = b
            self.lin[i] += v
            return
        i, j = min(i, j), max(i, j)
        self.quad[(i, j)] = self.quad.get((i, j), 0.0) + v

    def add_exactly_one(self, idx, lam):
        """lam * (1 - sum b)^2 : zero only when exactly one of the switches is ON."""
        self.const += lam
        for i in idx:
            self.lin[i] -= lam
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                self.add_quad(idx[a], idx[b], 2.0 * lam)

    def mean_energy(self):
        """Average energy over all 2^n bitstrings (= what random guessing gives)."""
        return self.const + self.lin.sum() / 2 + sum(self.quad.values()) / 4

    def energy_of(self, b):
        """Energy of one bitstring given as int (bit k = qubit k)."""
        e = self.const + sum(self.lin[i] for i in range(self.n) if (b >> i) & 1)
        e += sum(v for (i, j), v in self.quad.items() if (b >> i) & 1 and (b >> j) & 1)
        return float(e)

    def all_energies(self):
        """Energy table of all 2^n bitstrings (classical, NumPy). Used ONLY to validate the
        penalty weights once; the quantum part never uses this table."""
        n = self.n
        idx = np.arange(1 << n, dtype=np.uint32)
        bits = [((idx >> k) & 1).astype(np.uint8) for k in range(n)]
        E = np.full(1 << n, self.const)
        for i in range(n):
            if self.lin[i]:
                E += self.lin[i] * bits[i]
        for (i, j), v in self.quad.items():
            E += v * (bits[i] & bits[j])
        return E

    def to_ising(self):
        """b = (1 - Z)/2   ->   const + sum h_i Z_i + sum J_ij Z_i Z_j"""
        c, h, J = self.const, np.zeros(self.n), {}
        for i in range(self.n):
            c += self.lin[i] / 2
            h[i] -= self.lin[i] / 2
        for (i, j), v in self.quad.items():
            c += v / 4
            h[i] -= v / 4
            h[j] -= v / 4
            J[(i, j)] = J.get((i, j), 0.0) + v / 4
        return c, h, J

    def ising_operator(self):
        """Returns (SparsePauliOp without the constant, constant). <H_qubo> = <op> + constant."""
        c, h, J = self.to_ising()
        terms = [("Z", [i], float(v)) for i, v in enumerate(h) if v != 0.0]
        terms += [("ZZ", [i, j], float(v)) for (i, j), v in J.items() if v != 0.0]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=self.n), float(c)


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------
class ArcEncoding:
    """x_ij = 1 if the truck drives i -> j (single vehicle, depot = node 0).

      cost    : sum d_ij x_ij
      visit   : every place has exactly one road leaving and one road arriving
                (this also covers the paper's depot constraint)
      subtour : lam * x_ij x_ji   (forbids 2-place loops, as in the paper)
    With <= 4 customers (5 places) that removes every possible broken loop. With more
    customers longer broken loops are possible; decode() rejects them as infeasible.
    """
    name = "arc"

    def __init__(self, D, lam=2.0):
        self.D = np.asarray(D, float)
        n = self.n_nodes = len(D)
        self.scale = self.D.max()
        Dn = self.D / self.scale
        self.var = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.var[(i, j)] = len(self.var)
        self.N, self.lam = len(self.var), lam
        q = self.qubo = QUBO(self.N)
        for (i, j), v in self.var.items():
            q.lin[v] += Dn[i, j]
        for i in range(n):
            q.add_exactly_one([self.var[i, j] for j in range(n) if j != i], lam)   # one road out
            q.add_exactly_one([self.var[j, i] for j in range(n) if j != i], lam)   # one road in
        for i in range(n):
            for j in range(i + 1, n):
                q.add_quad(self.var[i, j], self.var[j, i], lam)

    def encode(self, route):
        b = 0
        for a, c in zip(route[:-1], route[1:]):
            b |= 1 << self.var[(a, c)]
        return b

    def decode(self, b):
        n = self.n_nodes
        succ, indeg = {}, [0] * n
        for (i, j), v in self.var.items():
            if (b >> v) & 1:
                if i in succ:
                    return None
                succ[i] = j
                indeg[j] += 1
        if len(succ) != n or any(d != 1 for d in indeg):
            return None
        route, cur = [0], 0
        for _ in range(n):
            cur = succ[cur]
            route.append(cur)
        if route[-1] != 0 or len(set(route[:-1])) != n:        # must be ONE loop through every place
            return None
        return route


class OrderEncoding:
    """x_ct = 1 if customer c is the t-th stop (t = 0..m-1, m = number of customers).

      cost    : d(depot, 1st stop) + sum d(c, c') x_ct x_c'(t+1) + d(last stop, depot)
      penalty : every customer in exactly one position, every position holds exactly one customer
    Every valid answer is automatically one single loop, so no subtour term is needed.
    """
    name = "order"

    def __init__(self, D, lam=2.0):
        self.D = np.asarray(D, float)
        self.n_nodes = len(D)
        m = self.m = self.n_nodes - 1
        self.N, self.lam = m * m, lam
        self.scale = self.D.max()
        Dn = self.D / self.scale
        q = self.qubo = QUBO(self.N)
        for c in range(1, m + 1):
            q.lin[self._v(c, 0)] += Dn[0, c]
            q.lin[self._v(c, m - 1)] += Dn[c, 0]
        for t in range(m - 1):
            for c in range(1, m + 1):
                for c2 in range(1, m + 1):
                    if c != c2:
                        q.add_quad(self._v(c, t), self._v(c2, t + 1), Dn[c, c2])
        for c in range(1, m + 1):
            q.add_exactly_one([self._v(c, t) for t in range(m)], lam)
        for t in range(m):
            q.add_exactly_one([self._v(c, t) for c in range(1, m + 1)], lam)

    def _v(self, c, t):
        return (c - 1) * self.m + t

    def encode(self, route):
        b = 0
        for t, c in enumerate(route[1:-1]):
            b |= 1 << self._v(c, t)
        return b

    def decode(self, b):
        order = []
        for t in range(self.m):
            on = [c for c in range(1, self.m + 1) if (b >> self._v(c, t)) & 1]
            if len(on) != 1:
                return None
            order.append(on[0])
        if len(set(order)) != self.m:
            return None
        return [0, *order, 0]


ENCODINGS = {"arc": ArcEncoding, "order": OrderEncoding}


def build_validated(kind, D, optimal_routes, opt_cost, lams=(1.5, 2.0, 3.0, 5.0, 8.0),
                    max_check_qubits=22, verbose=True):
    """Choose the smallest penalty weight whose QUBO ground state(s) are exactly the optimal
    routes (checked against brute force). Classical check, done once before any quantum run."""
    for lam in lams:
        enc = ENCODINGS[kind](D, lam=lam)
        if enc.N > max_check_qubits:
            if verbose:
                print(f"  [{kind}] {enc.N} qubits: too many to validate by enumeration, using lam={lam}")
            return enc
        E = enc.qubo.all_energies()
        ground = np.flatnonzero(E <= E.min() + 1e-9)
        ok = abs(E.min() - opt_cost / enc.scale) < 1e-9 and all(
            (r := enc.decode(int(g))) is not None and abs(route_cost(D, r) - opt_cost) < 1e-6
            for g in ground)
        if verbose:
            print(f"  [{kind}] penalty weight {lam}: {'valid' if ok else 'ground state wrong, trying larger'}")
        if ok:
            return enc
    raise RuntimeError("No penalty weight produced a QUBO whose ground state is the optimal route.")
