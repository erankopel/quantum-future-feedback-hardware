#!/usr/bin/env python3
"""
Rung 6 (register 10): the detuning fix, tested against its pre-registered
prediction with the window's last seconds.

Register 9 diagnosed the delay blocks' structure as an idle-accumulated
coherent rotation (10.6 kHz; 29.3 degrees at Delta-t = 6.4 us) that the
isotropic fit misreads as contraction. The v0.6 outlook pre-registered the
fix and the prediction: refocus the rotation inside the delay and the
isotropic model's integer returns where it is robust.

DESIGN. One atomic job, two arms interleaved per setting, both with the
same total 6.4 us idle on the message qubit:
  plain: barrier, delay(6.4 us), barrier            (register 7 block 3)
  echo : barrier, delay(3.2) X delay(3.2) X, barrier (net identity;
         refocuses static Z-detuning accumulated during the idle)
Same triple, seed, epoch. 1024 shots per circuit: 144 x 1024 = 147k shots,
predicted charge ~40 s against ~43 s remaining in the window. If the
estimate is wrong in the bad direction the job is refused or clips the
window's tail; nothing else is at risk.

PRE-REGISTERED PREDICTIONS (v0.6 outlook, on the record before this run):
  P1  eta(plain) - eta(echo) ~ +0.06: the rotation's fake contraction is
      removed by the echo.
  P2  the polar rotation angle of the echo arm collapses from ~29 degrees
      toward the native gate-associated 5-7 degrees.
  P3  on the echo arm the isotropic model's index equals the measured
      index (the integer is robust at this block); on the plain arm it
      does not, reproducing register 9.

GATE GEcho: per setting, the two arms' compiled circuits differ only in
delay and x ops, with exactly two extra x on the echo arm's message qubit.
"""
import os, sys, json, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import rung1_hardware_bracket as R

TRIPLE = (21, 22, 36)
SHOTS = 1024
HALF_DT = 800            # 3.2 us at dt = 4 ns (register 7's granularity)


def build_arm(prep, meas, f, l, echo):
    qc = QuantumCircuit(3, 1)
    for g in R.PREP[prep]:
        getattr(qc, g)(0)
    if f: qc.x(1)
    if l: qc.x(2)
    qc.unitary(Operator(R.U8), [2, 1, 0], label="round")
    qc.barrier(0)
    if echo:
        qc.delay(HALF_DT, 0, unit="dt"); qc.x(0)
        qc.delay(HALF_DT, 0, unit="dt"); qc.x(0)
    else:
        qc.delay(2 * HALF_DT, 0, unit="dt")
    qc.barrier(0)
    for g in R.BASIS[meas]:
        getattr(qc, g)(0)
    qc.measure(0, 0)
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")

    pubs, keys = [], []
    for prep in R.PREP:
        for meas in R.BASIS:
            for f in (0, 1):
                for l in (0, 1):
                    pair = [transpile(build_arm(prep, meas, f, l, e),
                                      backend=backend, optimization_level=3,
                                      initial_layout=list(TRIPLE),
                                      seed_transpiler=7) for e in (0, 1)]
                    ca = Counter(dict(pair[0].count_ops()))
                    cb = Counter(dict(pair[1].count_ops()))
                    for op in set(ca) | set(cb):
                        if op in ("x", "delay", "barrier"):
                            continue
                        assert ca[op] == cb[op], "GEcho FAILED: %r" % op
                    dx = cb.get("x", 0) - ca.get("x", 0)
                    assert dx == 2, "GEcho FAILED: x delta %d" % dx
                    pubs += pair
                    keys += [[0, prep, meas, f, l], [1, prep, meas, f, l]]
    print("GEcho PASS: arms differ only in delay/x, +2 x on echo arm")
    total = len(pubs) * SHOTS
    print("job: %d circuits x %d shots = %.0fk shots; predicted charge "
          "~%.0f s against ~43 s remaining"
          % (len(pubs), SHOTS, total / 1e3, 80 * total / 294912))
    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    sampler = SamplerV2(mode=backend)
    job = sampler.run(pubs, shots=SHOTS)
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   layout=list(TRIPLE), shots=SHOTS, keys=keys,
                   delay_us=6.4, predictions=["P1 d_eta ~ +0.06",
                                              "P2 angle collapses",
                                              "P3 isotropic integer returns "
                                              "on echo arm"]),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "rung6_job.json"), "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())


if __name__ == "__main__":
    main()
