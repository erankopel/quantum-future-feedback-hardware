#!/usr/bin/env python3
"""
Paper VII register 1: is there a law behind the index-fidelity table?

Rung 1 produced a monotone table mapping two-qubit gate error to the
entanglement-breaking index of the implemented channel. A table is not a
paper. This register tests whether one parameter explains it.

HYPOTHESIS (two parts).
  H1 (CONTRACTION). Gate noise acts on the round's affine Bloch pair as a
      near-uniform contraction toward the maximally mixed fixed point:
          A_noisy = (1 - eta) A_ideal,   c_noisy = (1 - eta) c_ideal,
      with eta linear in the two-qubit error eps and set by the compiled
      two-qubit count:  eta = g * n_2q * eps.
  H2 (SUFFICIENCY). The index of the noisy channel is reproduced, integer
      for integer, by the ONE-PARAMETER model channel (1 - eta) (A, c) of the
      IDEAL round. If so, the whole table collapses onto a single curve
      n_index(eta) computable from the ideal target alone, and the hardware
      measurement reduces to reading one number, eta.

Both are falsifiable here. H1 is tested by residual structure (is the noisy
pair really a scalar multiple of the ideal one, or does it have anisotropy
the model throws away?). H2 is tested by exact integer comparison.

float64 explorer throughout, labelled as such. Gated: the noiseless
tomography must reproduce the series bloch_pair before anything is reported.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "anc"))

import rung1_hardware_bracket as R                            # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke   # noqa: E402
R.FakeSherbrooke = FakeSherbrooke
from qiskit_aer import AerSimulator                           # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error   # noqa: E402

BASIS = ["cx", "rz", "sx", "x"]
EPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)


def noisy_pair(eps):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(eps, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(eps / 10, 1), ["sx", "x"])
    sim = AerSimulator(method="density_matrix", noise_model=nm,
                       basis_gates=BASIS)
    return R._tomography(sim, None, 0.0, None, None, basis=BASIS)


def index_of(A, c):
    _, first_ppt, _ = R.index_bracket(A, c, 0.0, nmax=400)
    return first_ppt


def main():
    Ai, ci = R.bloch_pair(R.U8, R.tau(R.P_HOLD))

    # ---- G0: the noiseless tomography must return the series pair
    A0, c0 = R.measure_round_noiseless()
    dA, dc = np.linalg.norm(A0 - Ai), np.linalg.norm(c0 - ci)
    print("G0 noiseless tomography vs series bloch_pair: "
          "||dA||=%.2e ||dc||=%.2e  [%s]"
          % (dA, dc, "PASS" if max(dA, dc) < 1e-10 else "FAIL"))
    assert max(dA, dc) < 1e-10, "G0 FAILED"

    n2q = R.compile_report(R.FakeSherbrooke())["two_qubit"]
    print("compiled two-qubit count per round: n_2q = %d\n" % n2q)

    print("H1: is the noisy pair a uniform contraction of the ideal pair?")
    print("%-9s %-11s %-11s %-11s %-9s %s"
          % ("eps", "eta_fit", "eta/(n2q eps)", "aniso resid", "||dc||/||c||",
             "index"))
    rows = []
    for eps in EPS:
        A, c = noisy_pair(eps)
        # least-squares scalar: eta minimising ||A - (1-eta) Ai||_F
        s = float(np.sum(A * Ai) / np.sum(Ai * Ai))
        eta = 1.0 - s
        resid = float(np.linalg.norm(A - s * Ai) / np.linalg.norm(Ai))
        sc = float(np.sum(c * ci) / np.sum(ci * ci))
        idx = index_of(A, c)
        rows.append(dict(eps=eps, eta=eta, g=eta / (n2q * eps), aniso=resid,
                         c_scale=sc, index=idx))
        print("%-9.0e %-11.5f %-11.4f %-11.3e %-9.5f %s"
              % (eps, eta, eta / (n2q * eps), resid, sc, idx))

    gs = [r["g"] for r in rows]
    print("\n  g = eta / (n_2q eps):  mean %.4f, spread [%.4f, %.4f]"
          % (np.mean(gs), min(gs), max(gs)))

    print("\nH2: does the ONE-PARAMETER model reproduce the index exactly?")
    print("%-9s %-9s %-11s %-11s %s"
          % ("eps", "eta", "index sim", "index model", "verdict"))
    agree = 0
    for r in rows:
        eta = r["eta"]
        Am, cm = (1 - eta) * Ai, (1 - eta) * ci
        im = index_of(Am, cm)
        ok = (im == r["index"])
        agree += ok
        r["index_model"] = im
        print("%-9.0e %-9.5f %-11s %-11s %s"
              % (r["eps"], eta, r["index"], im,
                 "exact" if ok else "MISS by %s" % (im - r["index"])))
    print("\n  H2: %d/%d exact" % (agree, len(rows)))

    json.dump(dict(label="float64 exploratory (register 1, Paper VII)",
                   n_2q=n2q, rows=rows),
              open(os.path.join(HERE, "contraction_register1_f64.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
