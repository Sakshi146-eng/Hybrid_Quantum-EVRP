"""
classical_baselines.py  -  the classical methods QAOA is compared against.

  brute_force : tries every order of customers. Exact. Also returns ALL optimal routes.
  ilp_mtz     : exact Integer Linear Program (arc variables + MTZ subtour elimination), solved with
                SciPy's HiGHS MILP solver. This is the "exact ILP" baseline used in the paper.
  ortools     : Google OR-Tools routing solver (guided local search heuristic).
  greedy      : nearest-neighbour heuristic (the paper's "Greedy Search").
"""
import itertools
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from vrp_qubo import route_cost


def brute_force(D, tol=1e-9):
    n = len(D)
    best, routes = np.inf, []
    for perm in itertools.permutations(range(1, n)):
        r = [0, *perm, 0]
        c = route_cost(D, r)
        if c < best - tol:
            best, routes = c, [r]
        elif abs(c - best) <= tol:
            routes.append(r)
    return best, routes


def greedy(D):
    n = len(D)
    route, left = [0], set(range(1, n))
    while left:
        nxt = min(left, key=lambda j: D[route[-1], j])
        route.append(nxt)
        left.remove(nxt)
    route.append(0)
    return route_cost(D, route), route


def ilp_mtz(D):
    """min sum d_ij x_ij ; one arc out / in per place ; MTZ: u_i - u_j + (n-1) x_ij <= n-2."""
    n = len(D)
    arcs = [(i, j) for i in range(n) for j in range(n) if i != j]
    na = len(arcs)
    nv = na + (n - 1)                                   # x_ij binary, then u_1..u_{n-1} continuous
    c = np.zeros(nv)
    for k, (i, j) in enumerate(arcs):
        c[k] = D[i, j]
    A, lo, hi = [], [], []
    for i in range(n):
        row = np.zeros(nv)
        for k, (a, b) in enumerate(arcs):
            if a == i:
                row[k] = 1
        A.append(row); lo.append(1); hi.append(1)       # one road out of i
        row = np.zeros(nv)
        for k, (a, b) in enumerate(arcs):
            if b == i:
                row[k] = 1
        A.append(row); lo.append(1); hi.append(1)       # one road into i
    for k, (i, j) in enumerate(arcs):
        if i != 0 and j != 0:
            row = np.zeros(nv)
            row[k] = n - 1
            row[na + i - 1] = 1
            row[na + j - 1] = -1
            A.append(row); lo.append(-np.inf); hi.append(n - 2)
    integrality = np.concatenate([np.ones(na), np.zeros(n - 1)])
    bounds = Bounds(np.concatenate([np.zeros(na), np.ones(n - 1)]),
                    np.concatenate([np.ones(na), np.full(n - 1, n - 1)]))
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=integrality, bounds=bounds)
    succ = {i: j for k, (i, j) in enumerate(arcs) if res.x[k] > 0.5}
    route, cur = [0], 0
    for _ in range(n):
        cur = succ[cur]
        route.append(cur)
    return route_cost(D, route), route


def ortools_solve(D, time_limit_s=1):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(D)
    scale = 1000
    Di = (np.asarray(D) * scale).round().astype(int)
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(mgr)
    cb = routing.RegisterTransitCallback(lambda a, b: int(Di[mgr.IndexToNode(a), mgr.IndexToNode(b)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = time_limit_s
    sol = routing.SolveWithParameters(p)
    idx, route = routing.Start(0), [0]
    while not routing.IsEnd(idx):
        idx = sol.Value(routing.NextVar(idx))
        route.append(mgr.IndexToNode(idx))
    return route_cost(D, route), route


def run_all(D):
    """Run every classical method, return {name: dict(cost, route, seconds)}."""
    out = {}
    methods = [("brute_force", lambda: (lambda r: (r[0], r[1][0]))(brute_force(D))),
               ("ilp_mtz", lambda: ilp_mtz(D)),
               ("greedy", lambda: greedy(D))]
    try:
        import ortools  # noqa: F401
        methods.insert(2, ("ortools", lambda: ortools_solve(D)))
    except ImportError:
        pass
    for name, fn in methods:
        t0 = time.time()
        cost, route = fn()
        out[name] = dict(cost=float(cost), route=route, seconds=time.time() - t0)
    return out
