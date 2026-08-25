#!/usr/bin/env python3
"""
Paper VII register 3: the shot budget, done properly.

Rung 1 priced the measurement with a single Hoeffding radius applied
uniformly to all twelve entries of the affine Bloch pair. That is rigorous
and very loose, in three separate ways:

  (i)   Hoeffding ignores the actual click probability. Most tomography
        settings sit far from p0 = 1/2, where the binomial is much tighter.
  (ii)  every entry got the worst-case radius, although each entry is a
        weighted average of eight settings and averaging shrinks the radius.
  (iii) the twelve radii were treated as independent worst cases even though
        they descend from the same 72 measured probabilities.

This register fixes (i) and (ii) and leaves (iii) as stated future work.
Method:

  - CLOPPER-PEARSON. For each of the 72 settings, the exact binomial
    two-sided interval on p0 at level 1 - delta/72 (Bonferroni over
    settings, so the joint statement holds at 1 - delta). Exact, not
    asymptotic, and valid at any N including small counts.
  - EXACT PROPAGATION. <P> = 2 p0 - 1 carries the interval directly.
    r_out[prep][i] is a convex combination of four settings with the tau(p)
    weights, so its radius is that weighted combination of theirs. A[i][j]
    and c[i] are half-sums of two such, so their radii follow exactly. No
    entry is charged more uncertainty than it actually carries.

Compared head to head against the uniform Hoeffding ball of Rung 1 at the
same joint confidence.

float64 explorer for the probabilities; the interval propagation through the
rounds is Arb 256-bit. Gated on the noiseless tomography before reporting.
"""
import os, sys, json, math
import numpy as np
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "anc"))

import rung1_hardware_bracket as R                            # noqa: E402
from qiskit_aer import AerSimulator                           # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error   # noqa: E402

BASIS = ["cx", "rz", "sx", "x"]
DELTA = 1e-3
N_SETTINGS = 72


def clopper_pearson_halfwidth(p, n, alpha):
    """Worst-case half-width of the exact binomial interval at count k = p*n.
    Uses the expected count; reported as the radius that would be obtained."""
    k = p * n
    lo = 0.0 if k <= 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k >= n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return max(p - lo, hi - p)


def probabilities(eps):
    """The 72 exact click probabilities p0 under the given gate error."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(eps, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(eps / 10, 1), ["sx", "x"])
    sim = AerSimulator(method="density_matrix", noise_model=nm,
                       basis_gates=BASIS)
    circuits, keys = [], []
    for prep in R.PREP:
        for meas in R.BASIS:
            for f in (0, 1):
                for l in (0, 1):
                    circuits.append(R.build(prep, meas, f, l)); keys.append(
                        (prep, meas, f, l))
    from qiskit import transpile
    tq = transpile(circuits, sim, basis_gates=BASIS, optimization_level=3,
                   seed_transpiler=7)
    for c in tq:
        c.save_probabilities_dict(qubits=[0], label="p")
    out = sim.run(tq).result()
    p0 = {}
    for i, k in enumerate(keys):
        d = out.data(i)["p"]
        p0[k] = sum(v for b, v in d.items()
                    if (int(b, 0) if isinstance(b, str) else int(b)) == 0)
    return p0


def pair_and_radii(p0, shots, alpha_per_setting):
    """(A, c) and their EXACT per-entry radii from the propagated intervals."""
    P = float(R.P_HOLD)
    w = {0: (1 + P) / 2, 1: (1 - P) / 2}
    exp, rad = {}, {}
    for k, p in p0.items():
        exp[k] = 2 * p - 1
        rad[k] = 2 * clopper_pearson_halfwidth(p, shots, alpha_per_setting)

    r_out, r_rad = {}, {}
    for prep in R.PREP:
        r_out[prep] = np.array([sum(w[f] * w[l] * exp[(prep, ms, f, l)]
                                    for f in (0, 1) for l in (0, 1))
                                for ms in ("x", "y", "z")])
        r_rad[prep] = np.array([sum(w[f] * w[l] * rad[(prep, ms, f, l)]
                                    for f in (0, 1) for l in (0, 1))
                                for ms in ("x", "y", "z")])
    c = (r_out["z+"] + r_out["z-"]) / 2
    c_rad = (r_rad["z+"] + r_rad["z-"]) / 2
    A = np.zeros((3, 3)); A_rad = np.zeros((3, 3))
    for j, (pp, pm) in enumerate((("x+", "x-"), ("y+", "y-"), ("z+", "z-"))):
        A[:, j] = (r_out[pp] - r_out[pm]) / 2
        A_rad[:, j] = (r_rad[pp] + r_rad[pm]) / 2
    return A, c, A_rad, c_rad


def index_bracket_perentry(A, c, A_rad, c_rad, nmax=200):
    """Same propagation as Rung 1 but with a per-entry radius."""
    from flint import arb
    Ab = [[arb(float(A[i][j]), float(A_rad[i][j])) for j in range(3)]
          for i in range(3)]
    cb = [arb(float(c[i]), float(c_rad[i])) for i in range(3)]
    Ac = [[arb(1) if i == j else arb(0) for j in range(3)] for i in range(3)]
    cc = [arb(0)] * 3
    last_npt = 0
    for r in range(1, nmax + 1):
        Ac = [[sum((Ab[i][k] * Ac[k][j] for k in range(3)), arb(0))
               for j in range(3)] for i in range(3)]
        cc = [sum((Ab[i][k] * cc[k] for k in range(3)), arb(0)) + cb[i]
              for i in range(3)]
        lo, hi = R.lmin_enclosure(Ac, cc)
        if hi.upper() < 0:
            last_npt = r
        elif lo.lower() > 0:
            return last_npt, r
        else:
            return last_npt, None
    return last_npt, None


def main():
    Ai, ci = R.bloch_pair(R.U8, R.tau(R.P_HOLD))
    A0, c0 = R.measure_round_noiseless()
    d = max(np.linalg.norm(A0 - Ai), np.linalg.norm(c0 - ci))
    print("G0 noiseless tomography vs series pair: %.2e  [%s]\n"
          % (d, "PASS" if d < 1e-10 else "FAIL"))
    assert d < 1e-10, "G0 FAILED"

    alpha = DELTA / N_SETTINGS          # Bonferroni over the 72 settings
    rows = []
    for eps in (3e-3, 1e-3, 1e-4):
        p0 = probabilities(eps)
        A_pt, c_pt, _, _ = pair_and_radii(p0, 10**12, alpha)
        _, n_impl, _ = R.index_bracket(A_pt, c_pt, 0.0, nmax=200)
        print("=== two-qubit error %.0e : implemented index %s ==="
              % (eps, n_impl))
        print("    %-10s %-13s %-13s %-13s %s"
              % ("shots", "Hoeffding rad", "CP rad (max)", "Hoeffding LB",
                 "Clopper-Pearson LB"))
        for shots in (10**5, 10**6, 10**7, 10**8):
            A, c, A_rad, c_rad = pair_and_radii(p0, shots, alpha)
            l_cp, f_cp = index_bracket_perentry(A, c, A_rad, c_rad)
            rad_h = R.hoeffding_radius(shots, DELTA)
            l_h, f_h, _ = R.index_bracket(A, c, rad_h, nmax=200)
            rows.append(dict(eps=eps, shots=shots, implemented=n_impl,
                             hoeffding_radius=rad_h,
                             cp_radius_max=float(A_rad.max()),
                             hoeffding_lower=l_h, cp_lower=l_cp))
            print("    %-10.0e %-13.3e %-13.3e %-13s %s  (+%d rounds)"
                  % (shots, rad_h, A_rad.max(), "index > %d" % l_h,
                     "index > %d" % l_cp, l_cp - l_h))
        print()

    json.dump(dict(label="float64 probabilities, Arb propagation "
                         "(register 3, Paper VII)",
                   delta=DELTA, settings=N_SETTINGS, rows=rows),
              open(os.path.join(HERE, "budget_register3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
