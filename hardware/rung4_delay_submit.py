#!/usr/bin/env python3
"""
Rung 4 (register 7): delay scaling, the idle-decay discriminator.

Register 5 measured eta = 0.1002 and the calibration snapshot explained a
quarter of it. Register 6 acquitted dephasing-type idle noise: decoupling
recovered nothing beyond its own pulse cost. What remains is relaxation-type
idle decay versus gate-level effects, and the discriminator is direct: dial
a deliberate idle Delta-t into the round and read d(eta)/d(Delta-t), the
message qubit's idle contraction rate per microsecond, from the slope.

DESIGN. One atomic Sampler job of 3 x 72 = 216 circuits, three blocks:
Delta-t = 0, 3.2 and 6.4 microseconds of delay on the MESSAGE qubit,
inserted between the round unitary and the measurement basis rotation,
bracketed by barriers so all three blocks compile with the identical
structure. The zero block is the in-job baseline, so the slope is
epoch-matched and immune to calibration drift between jobs; the zero
block's eta against register 5's 0.1002 is itself a drift measurement.
Same triple, seed and shots as registers 5 and 6 throughout.

GATE GDelay: across the three blocks, per setting, the compiled circuits
may differ ONLY in delay durations; every other op count must be identical.
Asserted before submission.

SCOPE. The after-round delay probes the message qubit's own idle decay
(T1-type and residual T2-type together, as they contract the Bloch vector).
Ancilla idling during the round, which degrades the bath before it is
consumed, is not probed by this design and is stated as such.

Usage:  python rung4_delay_submit.py          # dry run
        python rung4_delay_submit.py --yes    # submit
"""
import os, sys, json, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import rung1_hardware_bracket as R

TRIPLE = (21, 22, 36)
SHOTS = 4096
DT_US = (0.0, 3.2, 6.4)          # deliberate idle per block, microseconds


def build_delay(prep, meas, f, l, delay_dt):
    qc = QuantumCircuit(3, 1)
    for g in R.PREP[prep]:
        getattr(qc, g)(0)
    if f: qc.x(1)
    if l: qc.x(2)
    qc.unitary(Operator(R.U8), [2, 1, 0], label="round")
    qc.barrier(0)
    if delay_dt > 0:
        qc.delay(delay_dt, 0, unit="dt")
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
    dt = backend.dt
    gran = getattr(backend.target, "granularity", 16) or 16

    def us_to_dt(us):
        n = int(round(us * 1e-6 / dt))
        return max(0, (n // gran) * gran)

    blocks = [us_to_dt(u) for u in DT_US]
    print("dt = %.3e s, granularity %d; delay blocks (dt units): %s"
          % (dt, gran, blocks))
    print("actual delays (us): %s"
          % [round(b * dt * 1e6, 3) for b in blocks])

    circuits, keys = [], []
    for prep in R.PREP:
        for meas in R.BASIS:
            for f in (0, 1):
                for l in (0, 1):
                    for bi, d in enumerate(blocks):
                        circuits.append(build_delay(prep, meas, f, l, d))
                        keys.append([bi, prep, meas, f, l])
    tq = transpile(circuits, backend=backend, optimization_level=3,
                   initial_layout=list(TRIPLE), seed_transpiler=7)

    # ---- GDelay: per setting, blocks differ only in delay
    for s in range(72):
        base = None
        for bi in range(3):
            c = Counter(dict(tq[s * 3 + bi].count_ops()))
            c.pop("delay", None); c.pop("barrier", None)
            if base is None:
                base = c
            else:
                assert c == base, "GDelay FAILED at setting %d block %d" % (s, bi)
    print("GDelay PASS: blocks differ only in delay/barrier ops")

    print("job: 216 circuits x %d shots on %s, triple %s" %
          (SHOTS, backend.name, TRIPLE))
    print("estimate: ~3x the 80 s of a 72-circuit job, i.e. ~240 s QPU")
    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    sampler = SamplerV2(mode=backend)
    job = sampler.run(tq, shots=SHOTS)
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   layout=list(TRIPLE), shots=SHOTS, keys=keys,
                   blocks_dt=blocks, dt_seconds=dt,
                   blocks_us=[b * dt * 1e6 for b in blocks],
                   baseline_eta=0.10022, dd_eta=0.10947),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "rung4_job.json"), "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())


if __name__ == "__main__":
    main()
