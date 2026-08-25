#!/usr/bin/env python3
"""
Rung 11 (register 15): the virtual-Z fix, tested with a signed A/B/C.

WHY. Rung 10 (reduced 2Q GST on the (21,22) link) found the CZ's dominant
coherent error is H(IZ) = +2.40 deg: a Z-rotation on the ANCILLA, 4.3x the
design's resolution floor, where the message qubit's own H(ZI) was 0.05 deg
and rung 9 found the single-qubit gate set carries no Z-type error at all.
A residual single-qubit phase accompanying a CZ is the textbook signature of
an uncalibrated Stark shift, and it is removable in software by a virtual-Z.

THE TEST. A single-arm "apply the correction and see" cannot distinguish a
real fix from a sign-convention error or from drift. So this runs THREE arms,
interleaved per setting inside ONE atomic job:

  arm 0  plain                      (no correction)
  arm 1  rz(-theta) on the ancilla after every (21,22) CZ   [predicted fix]
  arm 2  rz(+theta) on the ancilla after every (21,22) CZ   [wrong sign]

PRE-REGISTERED PREDICTION. If the rung-10 diagnosis is right, the measured
polar rotation angle of the round's noise map moves in OPPOSITE directions in
arms 1 and 2, roughly symmetrically about arm 0, with arm 1 the smaller. A
null in both arms refutes the mechanism; a shift in the SAME direction in both
indicates drift or a timing artefact rather than the phase correction.

SCOPE, honestly. Rung 10 measured only the (21,22) link. The compiled round
also uses (21,36), whose CZ phase is unmeasured, so only part of the round's
CZ phase can be corrected here and the expected effect is correspondingly
partial. Measuring (21,36) (rung 12) is what would allow a full correction.

WHY rz IS THE RIGHT KNOB. On this backend rz is virtual: zero duration, no
pulse. So the three arms have IDENTICAL schedules and identical gate timing,
which removes the timing confound that made the rung-3 decoupling comparison
hard to read. GATE GVZ asserts exactly that.

GATE GVZ (asserted before submit, per setting):
  * the three arms differ ONLY in rz operations;
  * the number of inserted rz equals the number of (21,22) CZ gates;
  * arms 1 and 2 carry equal and opposite inserted angles;
  * the compiled duration is identical across the three arms.

Usage:  python rung11_vz_submit.py            # dry run
        python rung11_vz_submit.py --yes      # submit
"""
import os, sys, json, argparse, math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2
import submit_core as S

HERE = os.path.dirname(os.path.abspath(__file__))
M, F, L = S.TRIPLE                 # 21 message, 22 ancilla-F, 36 ancilla-L
THETA_DEG = 2.397                  # rung10 H(IZ) on the (21,22) CZ
SHOTS = 1024
A_PC, B_PS = 0.0121, 2.577e-4      # cost model, fitted on actual charges only


def cost(n, sh):
    return n * (A_PC + B_PS * sh)


def insert_vz(tq, theta_rad, link=(M, F), on=F):
    """Return a copy of tq with rz(theta) inserted on `on` after every CZ that
    acts on `link`. rz is virtual on this backend: zero duration."""
    out = tq.copy_empty_like()
    n_ins = 0
    for inst in tq.data:
        out.append(inst)
        if inst.operation.name in ("cz", "ecr", "cx"):
            qs = frozenset(tq.find_bit(q).index for q in inst.qubits)
            if qs == frozenset(link):
                out.rz(theta_rad, out.qubits[on])
                n_ins += 1
    return out, n_ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--theta-deg", type=float, default=THETA_DEG,
                    dest="theta_deg")
    args = ap.parse_args()
    theta = math.radians(args.theta_deg)

    backend = S.service().backend("ibm_kingston")
    base, keys = [], []
    for st in S.all_settings():
        base.append(S.build_tomo(*st)); keys.append(list(st))
    tq = transpile(base, backend=backend, optimization_level=3,
                   initial_layout=list(S.TRIPLE), seed_transpiler=S.SEED)

    # link census on the compiled round
    c0 = Counter()
    for inst in tq[0].data:
        if inst.operation.name in ("cz", "ecr", "cx"):
            c0[frozenset(tq[0].find_bit(q).index for q in inst.qubits)] += 1
    print("compiled round two-qubit census (first circuit):")
    for k, v in c0.items():
        print("   link %s : %d gates%s" % (sorted(k), v,
              "   <- measured by rung10" if k == frozenset((M, F)) else
              "   <- UNMEASURED (rung12 would fix)"))
    n_corr = c0[frozenset((M, F))]
    n_tot = sum(c0.values())
    print("   correctable here: %d of %d two-qubit gates (%.0f%%)"
          % (n_corr, n_tot, 100 * n_corr / n_tot))

    pubs, pubkeys = [], []
    for i in range(len(tq)):
        a0 = tq[i]
        a1, n1 = insert_vz(a0, -theta)
        a2, n2 = insert_vz(a0, +theta)
        # ---- GATE GVZ
        for arm, n_ins in ((a1, n1), (a2, n2)):
            ca, cb = Counter(dict(a0.count_ops())), Counter(dict(arm.count_ops()))
            for op in set(ca) | set(cb):
                if op == "rz":
                    continue
                assert ca[op] == cb[op], "GVZ FAILED: op %r changed" % op
            assert cb["rz"] - ca["rz"] == n_ins, "GVZ FAILED: rz count"
            assert n_ins == n_corr, "GVZ FAILED: expected %d insertions" % n_corr
        assert n1 == n2, "GVZ FAILED: arms differ in insertion count"
        pubs += [a0, a1, a2]
        pubkeys += [[0] + keys[i], [1] + keys[i], [2] + keys[i]]
    print("GVZ PASS: arms differ only in rz; %d insertions per circuit; "
          "arms 1/2 equal and opposite (%+.3f / %+.3f deg)"
          % (n_corr, -args.theta_deg, args.theta_deg))

    # GVZ timing: rz must be VIRTUAL (zero duration) or the arms differ in
    # schedule and the comparison is confounded. Check the target directly --
    # QuantumCircuit.duration returns None on qiskit 2.x, so comparing it is
    # a vacuous test (None == None) and must not be used as evidence.
    tgt = backend.target
    durs = {}
    for q in (M, F, L):
        try:
            durs[q] = tgt["rz"][(q,)].duration
        except Exception as e:
            durs[q] = "n/a"
    allzero = all(d == 0 for d in durs.values())
    print("GVZ timing: target rz durations %s -> %s"
          % (durs, "VIRTUAL, schedules identical" if allzero
             else "NOT virtual - arms would differ in timing, REVIEW"))
    assert allzero, "GVZ FAILED: rz is not virtual on this backend"

    n = len(pubs)
    c = cost(n, args.shots)
    print("\njob: %d circuits (3 arms x 72) x %d shots on %s, triple %s"
          % (n, args.shots, backend.name, S.TRIPLE))
    print("estimated QPU: %.0f s   (model fitted on actual charges, +/-2%%)" % c)
    u = S.service().usage()
    print("live usage: consumed %ss, remaining %ss -> after this run %ss"
          % (u["usage_consumed_seconds"], u["usage_remaining_seconds"],
             int(u["usage_remaining_seconds"] - c)))

    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    job = SamplerV2(mode=backend).run(pubs, shots=args.shots)
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   layout=list(S.TRIPLE), shots=args.shots, keys=pubkeys,
                   theta_deg=args.theta_deg, arms=["plain", "minus", "plus"],
                   n_corrected_gates=n_corr, n_twoq_total=n_tot,
                   est_cost_s=c),
              open(os.path.join(HERE, "rung11_job.json"), "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())


if __name__ == "__main__":
    main()
