"""
run_experiment.py  -  QUBO + QAOA (Qiskit) vs classical methods on a small VRP instance.

Examples
  python run_experiment.py --file E-n29-k4-s7.evrp --customers 3 --kind both --p 1 2
  python run_experiment.py --customers 4 --kind order --p 1 2          # random instance
"""
import argparse
import json
import random

from classical_baselines import brute_force, run_all
from qaoa_qiskit import evaluate_counts, run_qaoa
from vrp_qubo import (build_validated, dist_matrix, load_instance, random_instance,
                      route_cost, subsample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help=".evrp / TSPLIB-like .vrp file. Omit for a random instance")
    ap.add_argument("--customers", type=int, default=3)
    ap.add_argument("--select", default="nearest", choices=["nearest", "first", "random"])
    ap.add_argument("--kind", default="both", choices=["arc", "order", "both"])
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2], help="QAOA depths to try")
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    xy, ids = subsample(load_instance(a.file), a.customers, a.select, a.seed) if a.file \
        else random_instance(a.customers, a.seed)
    D = dist_matrix(xy)
    print(f"Instance: depot + {a.customers} customers, original node ids {ids}")

    # ---- classical methods -------------------------------------------------------------
    classical = run_all(D)
    opt, opt_routes = brute_force(D)
    print("\nCLASSICAL METHODS")
    print(f"{'method':<14}{'cost':>10}{'gap %':>9}{'time (s)':>11}")
    for name, r in classical.items():
        print(f"{name:<14}{r['cost']:>10.3f}{100 * (r['cost'] - opt) / opt:>9.2f}{r['seconds']:>11.4f}")

    # ---- QAOA --------------------------------------------------------------------------
    kinds = ["arc", "order"] if a.kind == "both" else [a.kind]
    results = dict(instance_ids=ids, optimal_cost=opt, classical=classical, qaoa=[])
    for kind in kinds:
        print(f"\nQAOA ({kind} encoding)")
        enc = build_validated(kind, D, opt_routes, opt)
        print(f"  qubits: {enc.N}, penalty weight: {enc.lam}")

        rng = random.Random(a.seed)                        # no-quantum reference: random bitstrings
        rand_counts = {}
        for _ in range(a.shots):
            b = rng.getrandbits(enc.N)
            rand_counts[b] = rand_counts.get(b, 0) + 1
        rs = evaluate_counts(enc, rand_counts, D, opt)
        print(f"  random bitstrings : feasible {rs['feasible_frac']:.3f}  optimal {rs['optimal_frac']:.4f}"
              f"  best cost {rs['best_cost']}")

        for p in a.p:
            res, _ = run_qaoa(enc, D, opt, opt_routes, p, a.shots, a.restarts, a.maxiter, a.seed)
            res.update(kind=kind, qubits=enc.N, penalty=enc.lam, random_baseline=rs)
            results["qaoa"].append(res)
            print(f"  QAOA p={p}: P(optimal route)={res['prob_optimal']:.4f} "
                  f"(x{res['boost_vs_uniform']:.0f} vs random) | shots: feasible {res['feasible_frac']:.3f}, "
                  f"optimal {res['optimal_frac']:.3f}, best cost {res['best_cost']} (gap {res['gap_pct']}%) | "
                  f"energy improvement {res['energy_improvement']:.2f} | {res['evaluations']} evals, "
                  f"{res['seconds']:.1f}s")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
