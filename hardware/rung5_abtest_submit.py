#!/usr/bin/env python3
"""
Rung 5 (register 8): the plain-versus-decoupling A/B, drift-proofed.

Register 7's in-job zero block exposed same-day drift of +0.023 in eta
across jobs, which confounds register 6's across-job DD comparison. This
job removes the confound the same way register 7 did: both arms inside ONE
Sampler job, interleaved per setting, so they share a calibration epoch.

Arms: (0) the plain tomography circuits of register 5; (1) the same
circuits with ALAP-scheduled XX decoupling, exactly as register 6. Gate
GDD asserts per setting that only x/delay counts differ. 144 circuits,
4096 shots, same triple and seed as every run today.

The number this yields is clean: delta_eta = eta_DD - eta_plain within one
epoch. Positive at the pulse-cost scale means decoupling costs more than it
recovers; negative means it recovers idle dephasing net of cost.
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

TRIPLE = (21, 22, 36)
SHOTS = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    service = QiskitRuntimeService()
    backend = service.backend("ibm_kingston")
    target = backend.target

    base, keys = [], []
    for prep in R.PREP:
        for meas in R.BASIS:
            for f in (0, 1):
                for l in (0, 1):
                    base.append(build_hw(prep, meas, f, l))
                    keys.append([prep, meas, f, l])
    tq = transpile(base, backend=backend, optimization_level=3,
                   initial_layout=list(TRIPLE), seed_transpiler=7)
    pm = PassManager([
        ALAPScheduleAnalysis(target=target),
        PadDynamicalDecoupling(target=target, dd_sequence=[XGate(), XGate()]),
    ])
    tq_dd = pm.run(tq)

    pubs, pubkeys = [], []
    for s_i in range(72):
        ca = Counter(dict(tq[s_i].count_ops()))
        cb = Counter(dict(tq_dd[s_i].count_ops()))
        for op in set(ca) | set(cb):
            if op in ("x", "delay", "barrier"):
                continue
            assert ca[op] == cb[op], "GDD FAILED at setting %d" % s_i
        pubs += [tq[s_i], tq_dd[s_i]]
        pubkeys += [[0] + keys[s_i], [1] + keys[s_i]]
    print("GDD PASS per setting; job: %d circuits x %d shots on %s"
          % (len(pubs), SHOTS, backend.name))
    print("estimate: ~2x 80 s = ~160 s QPU (ledger before: 399/600 used)")

    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    sampler = SamplerV2(mode=backend)
    job = sampler.run(pubs, shots=SHOTS)
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   layout=list(TRIPLE), shots=SHOTS, keys=pubkeys),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "rung5_job.json"), "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())


if __name__ == "__main__":
    main()
