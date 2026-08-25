#!/usr/bin/env python3
"""
Rung 10 (register 14): REDUCED two-qubit GST on the (21,22) CZ link.

WHY. Rung 9 characterised the message qubit's single-qubit gate set and found
coherent errors of only ~1 degree, essentially PURE on-axis under-rotation
with H(Z) ~ 0. That is a negative result for the paper's open question: the
5-9 degree gate-associated rotation that survives the rung-6 echo cannot be
blamed on single-qubit gate calibration. The remaining suspect is the CZ
layer -- the compiled round carries 24 two-qubit gates.

SCOPE. Register 14 ran this as a REDUCED design (L<=4, 1407 FPR circuits, 512
shots) on each link in turn, which gave an eigenphase of 4.15 +/- 1.54 deg on
(21,36) and 0.35 +/- 1.49 deg on (21,22): suggestive at 2.7 sigma, not
decisive. Window 3's W3-2 is the DEEPER design, L<=8, 1813 FPR circuits at 384
shots, ~207 s on the corrected price model. Doubling L roughly halves the
eigenphase uncertainty (1.5 -> ~0.8 deg). Defaults below are the W3-2 values;
raising --shots to 512 at L<=8 costs 272 s and does NOT fit the Path B ledger.

COMPILATION. Same pitfall as rung 9: germ structure must survive. Compiled at
optimization_level=0 with a barrier after every logical layer. GATE GGST2
asserts, circuit by circuit, that transpiled sx-count and cz-count equal the
logical counts, for every circuit, before anything is submitted.

BUDGET GUARD. The account has a hard rolling cap. This script queries live
usage before every chunk and refuses to submit a chunk that would take total
consumption past CAP_TOTAL. It reports exactly where it stopped.

Usage:  python rung10_gst2q_submit.py           # dry run
        python rung10_gst2q_submit.py --yes     # submit
"""
import os, sys, json, argparse
from math import pi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2
from pygsti.modelpacks import smq2Q_XYICPHASE as MP
import submit_core as S
import w3_cost as W

HERE = os.path.dirname(os.path.abspath(__file__))
PAIR = (21, 22)          # message qubit + first ancilla (the worse link)
MAXL = 8          # W3-2 as pre-registered: L<=8, 1813 FPR circuits
SHOTS = 384       # W3-2 as pre-registered; 512 would cost 272 s, not 207 s
CHUNK = 250
FPR = True        # fiducial pair reduction: 3703 -> 1407 circuits,
                  # drops redundant fiducial PAIRS, keeps L<=4 amplification
CAP_TOTAL = 500          # deliberately BELOW the plan's G-STAB cap of 550:
                         # W3-2 is the single biggest item in the window and
                         # this reserves 50 s of the cap for stability epochs


def gst_to_qiskit(c):
    """pyGSTi 2Q circuit -> qiskit, barrier after each logical layer."""
    qc = QuantumCircuit(2, 2)
    for i in range(c.depth):
        layer = c.layer(i)
        if len(layer) == 0:
            qc.id(0); qc.id(1)
        for lbl in layer:
            q = lbl.qubits
            if lbl.name == "Gxpi2":
                qc.sx(int(q[0]))
            elif lbl.name == "Gypi2":
                t = int(q[0])
                qc.rz(-pi / 2, t); qc.sx(t); qc.rz(pi / 2, t)
            elif lbl.name == "Gcphase":
                qc.cz(int(q[0]), int(q[1]))
            else:
                qc.id(0); qc.id(1)
        qc.barrier(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def logical_counts(c):
    sx = cz = 0
    for i in range(c.depth):
        for lbl in c.layer(i):
            if lbl.name in ("Gxpi2", "Gypi2"):
                sx += 1
            elif lbl.name == "Gcphase":
                cz += 1
    return sx, cz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--maxL", type=int, default=MAXL)
    ap.add_argument("--max-chunks", type=int, default=0,
                    dest="max_chunks", help="submit at most N chunks "
                    "(0 = all). Use 1 first to calibrate real cost.")
    ap.add_argument("--start-at", type=int, default=0, dest="start_at",
                    help="resume: first circuit index to submit")
    ap.add_argument("--pair", type=str, default=None,
                    help="physical pair, e.g. 21,36 (default 21,22)")
    args = ap.parse_args()

    pair = tuple(int(x) for x in args.pair.split(',')) if args.pair else PAIR
    edesign = MP.create_gst_experiment_design(args.maxL, fpr=FPR)
    circs = list(edesign.all_circuits_needing_data)
    qcs = [gst_to_qiskit(c) for c in circs]

    backend = S.service().backend("ibm_kingston")
    tq = transpile(qcs, backend=backend, optimization_level=0,
                   initial_layout=list(pair), seed_transpiler=S.SEED)

    bad = []
    for i, (c, t) in enumerate(zip(circs, tq)):
        lsx, lcz = logical_counts(c)
        ops = dict(t.count_ops())
        if ops.get("sx", 0) != lsx or ops.get("cz", 0) != lcz:
            bad.append(i)
    assert not bad, "GGST2 FAILED on %d circuits (e.g. %s)" % (len(bad), bad[:5])
    print("GGST2 PASS: all %d circuits preserve sx and cz counts" % len(tq))

    # Amendment 1a: the per-job intercept is paid once per CHUNK, so the
    # whole-design estimate must count chunks, not treat it as one job.
    def real_cost(n, sh, jobs=1): return W.cost(n, sh, n_jobs=jobs)
    n_chunks = (len(tq) + CHUNK - 1) // CHUNK
    total_cost = real_cost(len(tq), args.shots, jobs=n_chunks)
    print("2Q GST %s, L<=%d, pair %s" % ("smq2Q_XYICPHASE", args.maxL, pair))
    print("  %d circuits, max depth %d, %d shots"
          % (len(tq), max(t.depth() for t in tq), args.shots))
    print("  estimated total QPU: ~%.0f s in %d chunks of <=%d circuits "
          "(price from %s)"
          % (total_cost, (len(tq) + CHUNK - 1) // CHUNK, CHUNK, W.source()))

    svc = S.service()
    u = svc.usage()
    print("  live usage: consumed %ss, remaining %ss, cap for this run %ss"
          % (u["usage_consumed_seconds"], u["usage_remaining_seconds"], CAP_TOTAL))

    consumed = u["usage_consumed_seconds"]
    if consumed + total_cost > CAP_TOTAL:
        print("  UP-FRONT GUARD: %s s consumed + %.0f s for the whole design "
              "would pass the %s s cap. Reduce --maxL or --shots, or run it "
              "in the next window; a half-finished GST dataset is not usable."
              % (consumed, total_cost, CAP_TOTAL))
        if not args.yes:
            print("\nDRY RUN.")
        return
    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    sampler = SamplerV2(mode=backend)
    recfile = os.path.join(HERE, "rung10_job_%s_%d.json"
                           % ("".join(str(x) for x in pair), args.start_at))
    def record(jobs):
        json.dump(dict(start_at=args.start_at, backend=backend.name,
                       pair=list(pair), maxL=args.maxL, shots=args.shots,
                       modelpack="smq2Q_XYICPHASE", n_circuits=len(tq),
                       chunk=CHUNK, jobs=jobs,
                       circuit_strs=[c.str for c in circs],
                       submitted_circuits=sum(j["n"] for j in jobs)),
                  open(recfile, "w"), indent=1)
    chunks, jobs = [], []
    for start in range(args.start_at, len(tq), CHUNK):
        if args.max_chunks and len(jobs) >= args.max_chunks:
            print('  stopping: --max-chunks %d reached' % args.max_chunks)
            break
        chunk = tq[start:start + CHUNK]
        proj = real_cost(len(chunk), args.shots)
        # GUARD FIX (review, 25 Aug): sampler.run is fire-and-forget, so
        # svc.usage() cannot have billed the chunks already in flight. Without
        # the in-flight sum every iteration compared the SAME stale consumed
        # figure against ONE chunk and the guard never fired: 8 chunks could
        # bill ~700 s against a 600 s budget. In-flight estimates now count.
        inflight = sum(j["est_s"] for j in jobs)
        u = svc.usage()
        proj_total = u["usage_consumed_seconds"] + inflight + proj
        if proj_total > CAP_TOTAL:
            print("BUDGET GUARD: stopping before chunk at %d (consumed %ss "
                  "+ %.0fs in flight + %.0fs projected = %.0fs would pass cap %ss)"
                  % (start, u["usage_consumed_seconds"], inflight, proj,
                     proj_total, CAP_TOTAL))
            break
        job = sampler.run(chunk, shots=args.shots)
        jobs.append(dict(job_id=job.job_id(), start=start,
                         n=len(chunk), est_s=proj))
        chunks.append((start, len(chunk)))
        record(jobs)          # written after EVERY chunk, not only at the end
        print("  submitted chunk %d..%d : %s (est %.0f s)"
              % (start, start + len(chunk) - 1, job.job_id(), proj))

    record(jobs)
    print("\nsubmitted %d/%d circuits in %d job(s); wrote rung10_job_%s_%d.json"
          % (sum(j["n"] for j in jobs), len(tq), len(jobs),
             "".join(str(x) for x in pair), args.start_at))


if __name__ == "__main__":
    main()
