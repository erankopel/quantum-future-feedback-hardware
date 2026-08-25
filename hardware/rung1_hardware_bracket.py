#!/usr/bin/env python3
"""
Rung 1: an entanglement-breaking index bracket from ONE measured round.

The staircase experiment of the merged Paper V cannot be run to n = 88 rounds
on hardware; no device holds coherence that long. It does not have to be. The
loop's round map is a FIXED channel and the composite is its power, so one
round measured well determines the whole trajectory, PROVIDED the measurement
uncertainty is carried as an interval rather than a point estimate. That is
precisely what the series certification stack already does, so the hardware
result and the certified arithmetic meet without either giving ground.

PIPELINE

  1. COMPILE. The round unitary U(theta, phi, kappa, beta) on (M, F, L) is
     transpiled to a real IBM basis against a real coupling map. The gate
     counts printed here are Paper V's compilation table, measured rather
     than estimated.

  2. MEASURE. With the ancillas fixed at polarisation p, the round is a
     one-qubit channel on the message qubit, so its affine Bloch pair (A, c)
     is read off directly: prepare the six Pauli eigenstates, read <X>, <Y>,
     <Z> out, average over the ancilla bitstrings with the classical weights
     of tau(p). Twelve real parameters. Here this runs against a
     density-matrix simulation of a real backend's calibrated noise model,
     with that backend's readout error applied, so (A, c) is the NOISY round.

  3. BRACKET. Shot noise becomes a rigorous two-sided radius by Hoeffding:
     with N shots per setting and confidence 1 - delta per Pauli expectation,
         |<P>_hat - <P>|  <=  2 sqrt( ln(2/delta) / (2N) ),
     the factor 2 because Hoeffding bounds the click probability p0 and
     <P> = 2 p0 - 1.
     Every entry of (A, c) becomes an Arb ball of that radius.

  4. PROPAGATE. Balls are composed over r rounds in Arb 256-bit ball
     arithmetic and lmin of the partial-transposed Choi is enclosed at each
     r by the series primitives. The index is bracketed: the last r certified
     NPT is a lower bound, the first r certified PPT an upper bound. Once the
     ball has outgrown the margin the enclosure straddles zero and the round
     is UNRESOLVED. That is the honest outcome, and the thing the shot budget
     has to buy down.

EPISTEMIC LABEL. The output is NOT "certified" in the series sense, which
means certified by interval arithmetic alone. It is CERTIFIED-GIVEN-SHOTS:
arithmetically certified conditional on a statistical confidence. That is a
third epistemic category alongside certified and explorer, and it must be
labelled as one wherever it appears in a manuscript.

Requires: qiskit, qiskit-aer, qiskit-ibm-runtime, python-flint, numpy.
"""
import os, sys, json, math, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("PAPERV_ANC", ""),
             "/home/hermes-202/Documents/Arxiv/Paper-V-VI-merge/tex/anc",
             os.path.join(HERE, "..", "anc")):
    if cand and os.path.isfile(os.path.join(cand, "pulse_engine.py")):
        sys.path.insert(0, cand); break
sys.path.insert(0, HERE)

from flint import arb, acb                                    # noqa: E402
import pulse_engine as pe                                     # noqa: E402
from velocity_probe import U, QR1, HOLD, bloch_pair, tau      # noqa: E402

from qiskit import QuantumCircuit, transpile                  # noqa: E402
from qiskit.quantum_info import Operator                      # noqa: E402
from qiskit_aer import AerSimulator                           # noqa: E402
from qiskit_aer.noise import NoiseModel, depolarizing_error     # noqa: E402
from qiskit_ibm_runtime.fake_provider import (                # noqa: E402
    FakeSherbrooke, FakeLagosV2, FakePerth, FakeNairobiV2)

P_HOLD = float(HOLD)
U8 = U(QR1["theta"], QR1["phi"], QR1["kappa"], QR1["beta"])
DELTA = 1e-3                       # Hoeffding failure probability per Pauli

PREP = {"z+": [], "z-": ["x"], "x+": ["h"], "x-": ["x", "h"],
        "y+": ["h", "s"], "y-": ["x", "h", "s"]}
BASIS = {"x": ["h"], "y": ["sdg", "h"], "z": []}


def build(prep, meas, f, l):
    """One tomography circuit on (M, F, L) = virtual qubits (0, 1, 2)."""
    qc = QuantumCircuit(3)
    for g in PREP[prep]:
        getattr(qc, g)(0)
    if f: qc.x(1)
    if l: qc.x(2)
    # qiskit is little-endian: U8 = kron(M, F, L) puts M in the HIGHEST
    # operator qubit, so the qargs list is reversed. Gate G0 in main() checks
    # this against the series bloch_pair before any noisy number is reported.
    qc.unitary(Operator(U8), [2, 1, 0], label="round")
    for g in BASIS[meas]:
        getattr(qc, g)(0)
    return qc


def compile_report(backend, layout=None, opt_level=3):
    qc = build("z+", "z", 0, 0)
    t = transpile(qc, backend=backend, optimization_level=opt_level,
                  initial_layout=layout, seed_transpiler=7)
    ops = dict(t.count_ops())
    two = sum(v for k, v in ops.items() if k in ("ecr", "cz", "cx"))
    return dict(backend=backend.name, gates=ops, two_qubit=two, depth=t.depth())


def pick_triple(backend):
    """Best (message, anc1, anc2): both ancillas adjacent to the message
    qubit, chosen to minimise message-qubit readout error then link error."""
    tgt = backend.target
    edges = set()
    for name in ("ecr", "cz", "cx"):
        if name in tgt:
            edges |= set(tgt[name].keys())
    ro = {q: tgt["measure"][(q,)].error for q in range(backend.num_qubits)}
    link = {tuple(sorted(e)): tgt[[n for n in ("ecr", "cz", "cx")
                                   if n in tgt][0]][e].error for e in edges}
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    best = None
    for m, nb in adj.items():
        for f, l in itertools.combinations(sorted(nb), 2):
            cost = (ro[m], link[tuple(sorted((m, f)))] + link[tuple(sorted((m, l)))])
            if best is None or cost < best[0]:
                best = (cost, (m, f, l))
    return best[1], ro, link


def measure_round_noiseless():
    """G0: the same tomography with no noise must return the series pair."""
    sim = AerSimulator(method="density_matrix")
    return _tomography(sim, list(range(3)), 0.0, layout=None, backend=None)


def _tomography(sim, wires, p_ro, layout, backend, opt_level=3, basis=None):
    w = {0: (1 + P_HOLD) / 2, 1: (1 - P_HOLD) / 2}
    circuits, keys = [], []
    for prep in PREP:
        for meas in BASIS:
            for f in (0, 1):
                for l in (0, 1):
                    circuits.append(build(prep, meas, f, l))
                    keys.append((prep, meas, f, l))
    if backend is None:
        tq = transpile(circuits, sim, basis_gates=basis,
                       optimization_level=opt_level if basis else 1,
                       seed_transpiler=7)
        readout_wire = 0
    else:
        tq = transpile(circuits, backend=backend, optimization_level=opt_level,
                       initial_layout=layout, seed_transpiler=7)
        readout_wire = layout[0]
    for c in tq:
        c.save_probabilities_dict(qubits=[readout_wire], label="p")
    out = sim.run(tq).result()

    exp = {}
    for i, k in enumerate(keys):
        d = out.data(i)["p"]
        p0 = 0.0
        for b, v in d.items():
            bi = int(b, 0) if isinstance(b, str) else int(b)
            if bi == 0:
                p0 += v
        # symmetric readout confusion: <P>_observed = (1 - 2 p_ro) <P>_true
        exp[k] = (1 - 2 * p_ro) * (2 * p0 - 1)

    r_out = {p: np.array([sum(w[f] * w[l] * exp[(p, ms, f, l)]
                              for f in (0, 1) for l in (0, 1))
                          for ms in ("x", "y", "z")]) for p in PREP}
    c = (r_out["z+"] + r_out["z-"]) / 2
    A = np.zeros((3, 3))
    for j, (pp, pm) in enumerate((("x+", "x-"), ("y+", "y-"), ("z+", "z-"))):
        A[:, j] = (r_out[pp] - r_out[pm]) / 2
    return A, c


def measure_round_noiseless():
    """G0: the same tomography with no noise must return the series pair."""
    return _tomography(AerSimulator(method="density_matrix"), None, 0.0,
                       None, None)


def measure_round(backend, triple, opt_level=3):
    """Noisy affine Bloch pair (A, c) of one round: exact in the
    density-matrix simulation, with the backend's readout error applied."""
    noise = NoiseModel.from_backend(backend)
    sim = AerSimulator(method="density_matrix", noise_model=noise)
    return _tomography(sim, None, backend.target["measure"][(triple[0],)].error,
                       list(triple), backend, opt_level)


# ---------------------------------------------------------- Arb propagation
def compose_ball(A, c, rad, r):
    Ab = [[arb(float(A[i][j]), rad) for j in range(3)] for i in range(3)]
    cb = [arb(float(c[i]), rad) for i in range(3)]
    Ac = [[arb(1) if i == j else arb(0) for j in range(3)] for i in range(3)]
    cc = [arb(0)] * 3
    for _ in range(r):
        Ac = [[sum((Ab[i][k] * Ac[k][j] for k in range(3)), arb(0))
               for j in range(3)] for i in range(3)]
        cc = [sum((Ab[i][k] * cc[k] for k in range(3)), arb(0)) + cb[i]
              for i in range(3)]
    return Ac, cc


def lmin_enclosure(A, c):
    """Two-sided enclosure of lmin of the PT Choi from ball-valued (A, c)."""
    H = pe.choi_PT_pair(A, c)
    Hf = np.array([[complex(H[i, j].real.mid(), H[i, j].imag.mid())
                    for j in range(4)] for i in range(4)])
    _, V = np.linalg.eigh(Hf)
    v = [acb(float(V[i, 0].real), float(V[i, 0].imag)) for i in range(4)]
    num, den = acb(0), acb(0)
    for i in range(4):
        for j in range(4):
            num += v[i].conjugate() * H[i, j] * v[j]
        den += v[i].conjugate() * v[i]
    q = (num / den).real
    qm = q.mid()
    res2, nrm2 = acb(0), acb(0)
    for i in range(4):
        s = acb(0)
        for j in range(4):
            s += H[i, j] * v[j]
        d = s - acb(qm) * v[i]
        res2 += d * d.conjugate()
        nrm2 += v[i] * v[i].conjugate()
    rho = (res2.sqrt() / nrm2.sqrt()).real
    return arb(qm) - rho.upper(), arb(qm) + rho.upper()


def index_bracket(A, c, rad, nmax=200):
    """(last certified NPT, first certified PPT, first unresolved round)."""
    last_npt = 0
    for r in range(1, nmax + 1):
        Ar, cr = compose_ball(A, c, rad, r)
        lo, hi = lmin_enclosure(Ar, cr)
        if hi.upper() < 0:
            last_npt = r
        elif lo.lower() > 0:
            return last_npt, r, None
        else:
            return last_npt, None, r
    return last_npt, None, None


def hoeffding_radius(shots, delta=DELTA):
    """Radius on <P>, NOT on the click probability. Hoeffding bounds the
    Bernoulli estimate p0_hat by sqrt(ln(2/delta) / 2N); since
    <P> = 2 p0 - 1, the radius on <P> is twice that."""
    return 2 * math.sqrt(math.log(2 / delta) / (2 * shots))


def main():
    print("=" * 70)
    print("RUNG 1: index bracket from one measured round")
    print("=" * 70)

    print("\n[1] COMPILATION of one round (transpile only, no simulation)")
    for B in (FakeSherbrooke, FakeLagosV2):
        b = B()
        try:
            rep = compile_report(b)
        except Exception as e:
            print("    %s: %s" % (b.name, e)); continue
        print("    %-18s two-qubit %3d   depth %4d   %s"
              % (rep["backend"], rep["two_qubit"], rep["depth"], rep["gates"]))

    print("\n[G0] noiseless gate: tomography must return the series pair")
    Ai, ci = bloch_pair(U8, tau(P_HOLD))
    A0, c0 = measure_round_noiseless()
    dA, dc = np.linalg.norm(A0 - Ai), np.linalg.norm(c0 - ci)
    print("     ||dA||_F = %.3e   ||dc||_2 = %.3e   [%s]"
          % (dA, dc, "PASS" if max(dA, dc) < 1e-10 else "FAIL"))
    assert max(dA, dc) < 1e-10, "G0 FAILED - refusing to report noisy numbers"

    backend = FakeLagosV2()
    triple, ro, link = pick_triple(backend)
    print("\n[2] MEASUREMENT on %s (density-matrix, calibrated noise model)"
          % backend.name)
    print("    layout (M, F, L) = %s   readout err on M = %.4f"
          % (triple, ro[triple[0]]))
    A, c = measure_round(backend, triple)
    print("    ||A_noisy - A_ideal||_F = %.4e" % np.linalg.norm(A - Ai))
    print("    ||c_noisy - c_ideal||_2 = %.4e" % np.linalg.norm(c - ci))

    print("\n[3] INDEX, no shot noise (radius 0)")
    li, fi, _ = index_bracket(Ai, ci, 0.0)
    print("    ideal round  : last NPT %s, first PPT %s   (Paper V floor 89)"
          % (li, fi))
    ln, fn, un = index_bracket(A, c, 0.0)
    print("    measured round: last NPT %s, first PPT %s" % (ln, fn))

    print("\n[4] SHOT BUDGET   Hoeffding, delta = %.0e per Pauli expectation"
          % DELTA)
    print("    %-14s %-11s %-9s %-9s %s"
          % ("shots/setting", "radius", "last NPT", "first PPT", "verdict"))
    rows = []
    for shots in (10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10):
        rad = hoeffding_radius(shots)
        l, f, u = index_bracket(A, c, rad)
        verdict = ("index in (%d, %d]" % (l, f)) if f else \
                  ("UNRESOLVED from round %s" % u)
        rows.append(dict(shots=shots, radius=rad, last_npt=l,
                         first_ppt=f, first_unresolved=u))
        print("    %-14.0e %-11.2e %-9s %-9s %s"
              % (shots, rad, l, f if f else "-", verdict))

    print("\n[5] FIDELITY SPECIFICATION")
    print("    The device measures the index of the channel it ACTUALLY")
    print("    implements. Sweep two-qubit depolarizing error to find what")
    print("    a nontrivial index costs. (1-qubit error = eps/10, no readout.)")
    print("    %-12s %-11s %-11s %s" % ("2q error", "||dA||_F", "index", "verdict"))
    spec = []
    for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(eps, 2), ["cx", "ecr", "cz"])
        nm.add_all_qubit_quantum_error(depolarizing_error(eps / 10, 1), ["sx", "x"])
        sim = AerSimulator(method="density_matrix", noise_model=nm,
                           basis_gates=["cx", "rz", "sx", "x"])
        Ae, ce = _tomography(sim, None, 0.0, None, None, basis=["cx", "rz", "sx", "x"])
        le, fe, ue = index_bracket(Ae, ce, 0.0)
        d = float(np.linalg.norm(Ae - Ai))
        idx = fe if fe else "-"
        note = ("matches the ideal 88" if fe == 88 else
                ("within 5 of ideal" if fe and abs(fe - 88) <= 5 else
                 ("collapsed" if fe and fe < 20 else "off target")))
        spec.append(dict(eps=eps, dA=d, first_ppt=fe))
        print("    %-12.0e %-11.3e %-11s %s" % (eps, d, idx, note))

    print("\n[6] SHOT BUDGET at realistic gate error")
    print("    What the tomography can still resolve after r compositions.")
    budget = []
    for eps, lab in ((3e-3, "median Eagle link today"),
                     (1e-3, "best Eagle/Heron link"),
                     (1e-4, "one generation out")):
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(eps, 2), ["cx"])
        nm.add_all_qubit_quantum_error(depolarizing_error(eps / 10, 1), ["sx", "x"])
        sim = AerSimulator(method="density_matrix", noise_model=nm,
                           basis_gates=["cx", "rz", "sx", "x"])
        Ae, ce = _tomography(sim, None, 0.0, None, None,
                             basis=["cx", "rz", "sx", "x"])
        _, f0, _ = index_bracket(Ae, ce, 0.0)
        print("    2q error %.0e (%s): implemented index %s" % (eps, lab, f0))
        for shots in (10**6, 10**8, 10**9):
            rad = hoeffding_radius(shots)
            l, f, u = index_bracket(Ae, ce, rad)
            claim = ("index in (%d, %d]" % (l, f)) if f else \
                    ("certified lower bound: index > %d" % l)
            budget.append(dict(eps=eps, shots=shots, implemented=f0,
                               certified_lower=l, resolved=f))
            print("        %-10.0e shots/setting -> %s" % (shots, claim))

    json.dump(dict(backend=backend.name, layout=list(triple),
                   readout_error_M=ro[triple[0]],
                   A_measured=A.tolist(), c_measured=c.tolist(),
                   A_ideal=Ai.tolist(), c_ideal=ci.tolist(),
                   ideal_index=dict(last_npt=li, first_ppt=fi),
                   measured_index=dict(last_npt=ln, first_ppt=fn),
                   delta=DELTA, shot_budget=rows, fidelity_spec=spec,
                   realistic_budget=budget),
              open(os.path.join(HERE, "rung1_bracket.json"), "w"), indent=1)
    print("\n    wrote rung1_bracket.json")


if __name__ == "__main__":
    main()
