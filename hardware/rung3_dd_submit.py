#!/usr/bin/env python3
"""
Rung 3 (register 6): the dynamical-decoupling repeat.

The first run measured eta = 0.1002 on ibm_kingston and the calibration
snapshot explained only a quarter of it (links ~0.021, readout ~0.004). The
leading suspect for the 0.075 excess is decoherence during idle windows of
the depth-103 compiled round. Dynamical decoupling is the A/B test: repeat
the identical experiment with an XX decoupling sequence padded into every
idle window long enough to hold it, and read the change in eta.

CONTROLS. Everything that can be held fixed is held fixed: the same 72
tomography circuits, the same triple (21, 22, 36), the same optimisation
level and transpiler seed (so the same base circuits gate for gate), the
same 4096 shots, the same backend. The ONLY difference is the scheduling
pass: ALAPScheduleAnalysis + PadDynamicalDecoupling(XGate, XGate). Since
XX = identity, the logical circuit is unchanged; the transform may touch
only x-gate and delay counts, and the script asserts exactly that on every
circuit before it will submit.

READING THE OUTCOME (either sign is a result):
  eta_DD < eta_baseline: the difference is the DD-suppressible idle share
    (dephasing-type decoherence during idles), measured in eta units.
  eta_DD >= eta_baseline: the idle share is small and the added DD pulses
    cost more than they recover; the excess lives in coherent errors,
    leakage, or gate noise beyond the calibration snapshot. Also a result.

Usage:
    python rung3_dd_submit.py            # dry run: transform + accounting
    python rung3_dd_submit.py --yes      # submit
"""
import os, sys, json, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import transpile
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import rung1_hardware_bracket as R
from rung2_submit import build_hw

TRIPLE = (21, 22, 36)          # locked to the register-5 baseline
SHOTS = 4096                   # locked to the register-5 baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    target = backend.target

    circuits, keys = [], []
    for prep in R.PREP:
        for meas in R.BASIS:
            for f in (0, 1):
                for l in (0, 1):
                    circuits.append(build_hw(prep, meas, f, l))
                    keys.append([prep, meas, f, l])
    tq = transpile(circuits, backend=backend, optimization_level=3,
                   initial_layout=list(TRIPLE), seed_transpiler=7)

    pm = PassManager([
        ALAPScheduleAnalysis(target=target),
        PadDynamicalDecoupling(target=target, dd_sequence=[XGate(), XGate()]),
    ])
    tq_dd = pm.run(tq)

    # ---- gate GDD: the transform may change ONLY x and delay counts
    added_x = 0
    for a, b in zip(tq, tq_dd):
        ca, cb = Counter(dict(a.count_ops())), Counter(dict(b.count_ops()))
        for op in set(ca) | set(cb):
            if op in ("x", "delay", "barrier"):
                continue
            assert ca[op] == cb[op], \
                "GDD FAILED: op %r changed %d -> %d" % (op, ca[op], cb[op])
        dx = cb.get("x", 0) - ca.get("x", 0)
        assert dx >= 0 and dx % 2 == 0, "GDD FAILED: odd x insertion %d" % dx
        added_x += dx
    print("GDD PASS: only x/delay altered; %d X gates added over 72 circuits "
          "(mean %.1f/circuit)" % (added_x, added_x / 72))

    # ---- idle accounting on the first circuit (representative)
    dt = backend.dt
    idle = {q: 0 for q in TRIPLE}
    for inst in tq[0].data:
        if inst.operation.name == "delay":
            q = tq[0].find_bit(inst.qubits[0]).index
            if q in idle:
                d = inst.operation.duration
                idle[q] += d
    # scheduled durations only exist post-analysis; report from DD circuit
    idle_dd = {q: 0 for q in TRIPLE}
    for inst in tq_dd[0].data:
        if inst.operation.name == "delay":
            q = tq_dd[0].find_bit(inst.qubits[0]).index
            if q in idle_dd:
                idle_dd[q] += inst.operation.duration
    dur = getattr(tq_dd[0], "duration", None)
    if dur and dt:
        print("first circuit: total %.2f us; residual idle per qubit (us): %s"
              % (dur * dt * 1e6,
                 {q: round(v * dt * 1e6, 2) for q, v in idle_dd.items()}))

    print("job: 72 circuits x %d shots on %s, triple %s (identical to "
          "register 5 except the DD padding)" % (SHOTS, backend.name, TRIPLE))
    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    sampler = SamplerV2(mode=backend)
    job = sampler.run(tq_dd, shots=SHOTS)
    ro = target["measure"][(TRIPLE[0],)].error
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   layout=list(TRIPLE), shots=SHOTS, keys=keys,
                   dd="ALAP + XX", added_x_total=added_x,
                   calibration={"readout_err_M": ro},
                   baseline_job="da5vbee1vhnc73fmj5i0",
                   baseline_eta=0.10022, baseline_index=11),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "rung3_job.json"), "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())


if __name__ == "__main__":
    main()
