#!/usr/bin/env python3
"""
Rung 2, part 1: submit the single-round tomography to real IBM hardware.

WHAT THIS COSTS. The IBM Open Plan includes 10 minutes of QPU time per
28-day rolling window (plus, since 16 March 2026, an opt-in promotion of 180
additional minutes over 12 months). One run of this script is 72 circuits x
SHOTS shots in a single Sampler job; at the default 8192 shots that is about
0.6M shots total, of the order of one to two QPU minutes. The script prints
its estimate and refuses to submit without --yes.

WHAT THIS NEEDS. An IBM Quantum API token, either already saved on this
machine (QiskitRuntimeService.save_account) or in the environment variable
IBM_QUANTUM_TOKEN. Nothing else. Usage:

    python rung2_submit.py                 # dry run: plan + estimate only
    python rung2_submit.py --yes           # submit
    python rung2_submit.py --yes --shots 32768 --backend ibm_torino

WHAT IT DOES. Selects the least-busy operational backend (>= 3 qubits),
picks the (message, ancilla, ancilla) triple that minimises message-qubit
readout error then link error, transpiles the 72 tomography circuits of the
Paper VII pipeline at optimisation level 3 onto that triple, submits ONE
SamplerV2 job, and writes rung2_job.json with the job id, the backend name,
the layout, the calibration snapshot for the triple, and the compiled
two-qubit counts. rung2_analyze.py picks it up from there.

Epistemic note for the record: the tomography is linear inversion, so the
eta this run yields includes state preparation and measurement error along
with the round's own; the resulting index is the index of the implemented
round-plus-SPAM channel. That is stated in the manuscript and must be stated
with any number this run produces.
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import rung1_hardware_bracket as R


def build_hw(prep, meas, f, l):
    """Tomography circuit with a real measurement (hardware version)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(3, 1)
    for g in R.PREP[prep]:
        getattr(qc, g)(0)
    if f: qc.x(1)
    if l: qc.x(2)
    qc.unitary(Operator(R.U8), [2, 1, 0], label="round")   # little-endian: G0-checked order
    for g in R.BASIS[meas]:
        getattr(qc, g)(0)
    qc.measure(0, 0)
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually submit")
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--backend", type=str, default=None)
    args = ap.parse_args()

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
               if token else QiskitRuntimeService())

    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False,
                                     min_num_qubits=3)
    triple, ro, link = R.pick_triple(backend)
    print("backend : %s (%d qubits)" % (backend.name, backend.num_qubits))
    print("layout  : (M, F, L) = %s   readout err on M = %.4f"
          % (triple, ro[triple[0]]))

    circuits, keys = [], []
    for prep in R.PREP:
        for meas in R.BASIS:
            for f in (0, 1):
                for l in (0, 1):
                    circuits.append(build_hw(prep, meas, f, l))
                    keys.append([prep, meas, f, l])
    tq = transpile(circuits, backend=backend, optimization_level=3,
                   initial_layout=list(triple), seed_transpiler=7)
    ops = dict(tq[0].count_ops())
    two = sum(v for k, v in ops.items() if k in ("ecr", "cz", "cx"))
    total_shots = 72 * args.shots
    print("compiled: %d two-qubit gates, depth %d (first circuit)"
          % (two, tq[0].depth()))
    print("job     : 72 circuits x %d shots = %.2e shots total"
          % (args.shots, total_shots))
    print("estimate: roughly 1-3 QPU minutes at this size; Open Plan window "
          "is 10 min per 28 days (plus the 2026 opt-in promotion).")

    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    sampler = SamplerV2(mode=backend)
    job = sampler.run(tq, shots=args.shots)
    cal = {"readout_err_M": ro[triple[0]],
           "links": {str(k): v for k, v in link.items()
                     if triple[0] in k}}
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   layout=list(triple), shots=args.shots, keys=keys,
                   calibration=cal, two_qubit=two, depth=tq[0].depth()),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "rung2_job.json"), "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())
    print("wrote rung2_job.json; run rung2_analyze.py when the job completes.")


if __name__ == "__main__":
    main()
