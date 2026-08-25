#!/usr/bin/env python3
"""
Rung 7 (register 11): the segmented-idle / CPMG-order sweep.

WHAT IT CLOSES. The v0.7 outlook: "A segmented-delay run would close the
in-round idle picture." Inside the compiled round the message qubit's ~1.85 us
of idle is broken into segments BY the round's gates, and the paper's
hypothesis is that this segmented idle "contributes far less than the
contiguous-idle slope would suggest" because the interspersed operations
refocus the quasi-static detuning that register 9 identified (10.6 kHz).
Bare barrier-segmentation of a delay is physically identical to a contiguous
delay, so it tests nothing; the faithful probe is to sweep the NUMBER of
refocusing operations across a FIXED total idle and watch the coherent
rotation refocus.

DESIGN. One atomic Sampler job. Fixed total idle T = 6.4 us on the message
qubit, inserted between the round unitary and the basis rotation (same point
as rung4/rung6), filled as a CPMG sequence of n in {0, 2, 4, 8} evenly-spaced
X pulses (EVEN so the pulse train is net-identity and every block measures the
SAME logical channel). n=0 is the contiguous bare delay (today's epoch analog
of rung4's dt=6.4 block); n>0 refocuses. All four blocks share triple, seed
and epoch. 4 blocks x 72 = 288 circuits.

PREDICTIONS (register 9 mechanism, on the record before the run):
  Q1 the polar rotation angle FALLS with n: bare shows the full detuning
     rotation, CPMG refocuses it (angle(4) << angle(0)).
  Q2 eta' (contraction after the rotation is divided out) is roughly FLAT
     in n: it is the genuine incoherent decay, which refocusing does not
     touch. Any residual fall in eta at high n is the quasi-static dephasing
     share the echo also removes.
  Q3 the measured index RISES with n as the fake contraction is refocused.

GATE GSEG (asserted before submit): per setting, the four blocks' compiled
circuits differ ONLY in x/delay/barrier counts; every X-pulse-count delta
equals the block's n; every block's total inserted idle equals T to within
one granularity step.

Usage:  python rung7_seg_submit.py            # dry run (default)
        python rung7_seg_submit.py --yes      # submit
"""
import os, sys, json, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2
import submit_core as S

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_US = 6.4
# EVEN pulse counts only: an odd number of X pulses leaves a net X between the
# round and the read-out, which changes the logical channel and breaks the
# apples-to-apples comparison. n=0 is bare (contiguous), n>=2 are CPMG orders
# whose XX...X is net-identity, so every block measures round + refocused idle.
N_PULSES = (0, 2, 4, 8)
SHOTS = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--shots", type=int, default=SHOTS)
    args = ap.parse_args()

    service = S.service()
    backend = service.backend("ibm_kingston")
    dt = backend.dt
    gran = getattr(backend.target, "granularity", 16) or 16
    total_dt = int(round(TOTAL_US * 1e-6 / dt / gran)) * gran
    print("dt=%.2e s, gran=%d; total idle T=%.3f us = %d dt"
          % (dt, gran, total_dt * dt * 1e6, total_dt))

    # build logical circuits, grouped per setting so GSEG compares within setting
    logical, keys, xadd, idle = [], [], [], []
    for (prep, meas, f, l) in S.all_settings():
        for n in N_PULSES:
            qc, xa, tot = S.build_cpmg(prep, meas, f, l, total_dt, n, gran)
            logical.append(qc); keys.append([n, prep, meas, f, l])
            xadd.append(xa); idle.append(tot)

    # GSEG at the logical level (pre-transpile): within each setting, blocks
    # differ only in x/delay/barrier, x-delta == n, idle == T +/- gran
    nb = len(N_PULSES)
    for s in range(0, len(logical), nb):
        base = Counter(dict(logical[s].count_ops()))
        base_x = base.get("x", 0)
        for j, n in enumerate(N_PULSES):
            c = Counter(dict(logical[s + j].count_ops()))
            for op in set(c) | set(base):
                if op in ("x", "delay", "barrier"):
                    continue
                assert c[op] == base[op], "GSEG FAIL op %r setting %d" % (op, s)
            assert c.get("x", 0) - base_x == n, "GSEG FAIL x-delta"
            assert abs(idle[s + j] - total_dt) <= gran, "GSEG FAIL idle total"
    print("GSEG PASS (logical): blocks differ only in x/delay/barrier; "
          "x-delta == n; total idle == T within one gran step")

    tq = transpile(logical, backend=backend, optimization_level=3,
                   initial_layout=list(S.TRIPLE), seed_transpiler=S.SEED)

    # report compiled structure of the four blocks of the first setting
    print("compiled first-setting blocks (n : two-qubit, depth, x, delay):")
    twoq_names = ("ecr", "cz", "cx")
    for j, n in enumerate(N_PULSES):
        ops = dict(tq[j].count_ops())
        two = sum(v for k, v in ops.items() if k in twoq_names)
        print("   n=%d : 2q=%d depth=%d x=%d delay=%d"
              % (n, two, tq[j].depth(), ops.get("x", 0), ops.get("delay", 0)))

    n_circ = len(tq)
    cost = S.cost_seconds(n_circ, args.shots)
    print("job: %d circuits x %d shots on %s, triple %s"
          % (n_circ, args.shots, backend.name, S.TRIPLE))
    print("estimated QPU cost: ~%.0f s" % cost)

    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit.")
        return

    sampler = SamplerV2(mode=backend)
    job = sampler.run(tq, shots=args.shots)
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   layout=list(S.TRIPLE), shots=args.shots, keys=keys,
                   n_pulses=list(N_PULSES), total_idle_dt=total_dt,
                   total_idle_us=total_dt * dt * 1e6, dt_seconds=dt,
                   est_cost_s=cost),
              open(os.path.join(HERE, "rung7_job.json"), "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())


if __name__ == "__main__":
    main()
