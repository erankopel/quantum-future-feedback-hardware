#!/usr/bin/env python3
"""
Rung 9 (register 13): gate-set tomography on the message qubit.

WHAT IT CLOSES. Two things the manuscript flags as open:
  * "Not claimed: SPAM-free." Linear-inversion tomography folds prep and
    measurement error into eta. GST estimates the SPAM (rho, E) separately,
    so for the first time the series can quote how much of eta is SPAM.
  * "what survives the echo is the gate-associated rotation of 5 to 9
    degrees ... removing that needs calibration at the gate level, with
    gate-set tomography the natural instrument." GST returns each native
    gate's error generator, whose Hamiltonian (coherent) part is exactly the
    over-rotation / axis-tilt that compounds into the round's rotation.

SCOPE, stated honestly. This is 1-qubit long-sequence GST on the message
qubit's native single-qubit gate set {Gxpi2, Gypi2} (X/Y pi/2), L up to 8.
It characterises the SINGLE-QUBIT coherent errors and SPAM of qubit 21, not
the full three-qubit round (that is the genuine next-window object). The idle
detuning is left to rung6/rung7, which probe the ACTUAL round idle; GST here
isolates the gate-associated share and the SPAM.

COMPILATION, the pitfall handled. Long-sequence GST only works if each logical
gate maps to a fixed native pulse and germs are NOT merged. optimization_level
>= 1 merges sx.sx across barriers and destroys the germ (verified: a 6-sx germ
collapses to 1-4 sx). This script compiles at optimization_level=0 with a
barrier after every logical gate, and GATE GGST asserts, circuit by circuit,
that the transpiled sx-count equals the logical single-qubit-gate count for
all circuits before it will submit.

Usage:  python rung9_gst_submit.py            # dry run (default)
        python rung9_gst_submit.py --yes      # submit
        python rung9_gst_submit.py --maxL 16  # deeper (sharper angles)
"""
import os, sys, json, argparse
from math import pi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2
from pygsti.modelpacks import smq1Q_XY
import submit_core as S

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS = S.TRIPLE[0]            # message qubit 21
SHOTS = 1024
MAXL = 8


def gst_to_qiskit(c):
    """One pyGSTi circuit -> a 1-qubit qiskit circuit, barrier after each
    logical gate so opt_level=0 cannot merge germs."""
    qc = QuantumCircuit(1, 1)
    for i in range(c.depth):
        layer = c.layer(i)
        if len(layer) == 0:
            qc.id(0)
        for lbl in layer:
            if lbl.name == "Gxpi2":
                qc.sx(0)
            elif lbl.name == "Gypi2":
                qc.rz(-pi / 2, 0); qc.sx(0); qc.rz(pi / 2, 0)
            else:
                qc.id(0)
        qc.barrier(0)
    qc.measure(0, 0)
    return qc


def logical_sx(c):
    return sum(1 for i in range(c.depth) for lbl in c.layer(i)
               if lbl.name in ("Gxpi2", "Gypi2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--maxL", type=int, default=MAXL)
    ap.add_argument("--shots", type=int, default=SHOTS)
    args = ap.parse_args()

    edesign = smq1Q_XY.create_gst_experiment_design(args.maxL)
    circs = list(edesign.all_circuits_needing_data)
    qcs = [gst_to_qiskit(c) for c in circs]

    backend = S.service().backend("ibm_kingston")
    tq = transpile(qcs, backend=backend, optimization_level=0,
                   initial_layout=[PHYS], seed_transpiler=S.SEED)

    # GATE GGST: germ structure preserved for every circuit
    bad = [i for i, (c, t) in enumerate(zip(circs, tq))
           if dict(t.count_ops()).get("sx", 0) != logical_sx(c)]
    assert not bad, "GGST FAILED: %d circuits lost germ structure" % len(bad)
    print("GGST PASS: all %d circuits preserve germ structure (opt0+barrier)"
          % len(tq))

    print("GST design: smq1Q_XY, L<=%d, qubit %d" % (args.maxL, PHYS))
    print("  %d circuits, max depth %d"
          % (len(tq), max(t.depth() for t in tq)))
    cost = S.cost_seconds(len(tq), args.shots)
    print("  %d shots -> ~%.0f s QPU" % (args.shots, cost))

    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    sampler = SamplerV2(mode=backend)
    job = sampler.run(tq, shots=args.shots)
    # persist the edesign so the analyzer can rebuild the DataSet in order
    edir = os.path.join(HERE, "rung9_edesign")
    try:
        edesign.write(edir)
    except Exception as e:
        print("warn: edesign.write:", str(e)[:120])
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   qubit=PHYS, maxL=args.maxL, shots=args.shots,
                   modelpack="smq1Q_XY", n_circuits=len(tq),
                   circuit_strs=[c.str for c in circs], est_cost_s=cost),
              open(os.path.join(HERE, "rung9_job.json"), "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())


if __name__ == "__main__":
    main()
