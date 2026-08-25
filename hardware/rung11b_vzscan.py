#!/usr/bin/env python3
"""
Rung 11b (register 15): the virtual-Z correction as a five-arm angle scan.
Pre-registered for window 3. Amendment 1 (25 Aug 2026): the window opened on
25 Aug on a fresh 600 s instance, not 21 Sep; the estimators, the G-VZ gate and
V2's sign clause are amended in window3_plan_v2.txt and w3_vfit.py. Do not
submit before the W3-0 preflight passes.

WHY A SCAN, NOT AN A/B. Register 14 localised the round's gate-associated
coherent rotation to the (21,36) CZ (gauge-invariant eigenphase 4.2 +/- 1.5
degrees per gate); its per-component indication, a Z-phase of -2.84 degrees
on the message qubit, is gauge-sensitive AND a month stale by window 3, on a
device whose idle detuning doubled within a single day. A single-theta test
would test a stale number. The scan applies rz(k * theta0) on the message
qubit after every (21,36) CZ, k in {0, -1, +1, -2, +2}, theta0 = 2.84
degrees, all five arms interleaved per setting in ONE atomic job. rz is
virtual (zero duration, GVZ-asserted), so all arms share an identical
schedule.

PRE-REGISTERED PREDICTIONS (on the record before the run):
  V1 the measured polar rotation angle of the round's noise map is V-shaped
     in k with its minimum displaced from k = 0;
  V2 the minimum's angle is below the plain arm's by at least 3 sigma of
     the shot-noise floor. AMENDED A10: the DIRECTION is an output, not an
     assumption; the derived sign is re-issued as the side-prediction
     V2P, the minimum lies at k = +1;
  V3 eta (isotropic) falls at the minimum while eta' (rotation-corrected)
     is unchanged across arms within noise: the correction removes
     rotation, not decay;
  V4 at the minimum the isotropic model's integer agrees with the measured
     integer (the register-9 restoration, now achieved in-loop rather than
     in analysis).
Failure modes are informative: a flat response refutes the localisation; a
same-direction shift in both signed arms indicates drift, not phase.

GATE GVZ (asserted before submit): arms differ only in rz ops; per arm the
insertion count equals the number of (21,36) CZ gates in the compiled round;
k and -k arms carry equal and opposite total inserted angle; rz duration is
0 on all three qubits in the live target.

Usage:  python rung11b_vzscan.py            # dry run (default)
        python rung11b_vzscan.py --yes      # submit (window 3 only)
        python rung11b_vzscan.py analyze    # fetch + V-fit + w3_theta.json
"""
import os, sys, json, math, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2
import submit_core as S
import paper7_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
M = S.TRIPLE[0]                 # message qubit 21
LINK = frozenset((21, 36))      # the localised carrier
THETA0_DEG = 2.84
KS = (0, -1, 1, -2, 2)
SHOTS = 1024
import w3_cost as W
SAFETY = W.SAFETY
CAP_TOTAL = 550                 # G-STAB hard cap on projected consumption


def cost(n, sh):
    return W.cost(n, sh, loaded=False)


def insert_vz(tq, theta_rad, link=LINK):
    out = tq.copy_empty_like()
    n_ins = 0
    for inst in tq.data:
        out.append(inst)
        if inst.operation.name in ("cz", "ecr", "cx"):
            qs = frozenset(tq.find_bit(q).index for q in inst.qubits)
            if qs == link:
                if theta_rad != 0.0:
                    out.rz(theta_rad, out.qubits[M])
                n_ins += 1
    return out, n_ins


def build_arms(backend, theta0_deg, link=LINK, ks=KS):
    base, keys = [], []
    for st in S.all_settings():
        base.append(S.build_tomo(*st)); keys.append(list(st))
    tq = transpile(base, backend=backend, optimization_level=3,
                   initial_layout=list(S.TRIPLE), seed_transpiler=S.SEED)
    n_link = sum(1 for inst in tq[0].data
                 if inst.operation.name in ("cz", "ecr", "cx")
                 and frozenset(tq[0].find_bit(q).index
                               for q in inst.qubits) == link)
    pubs, pubkeys = [], []
    for i in range(len(tq)):
        arms = {}
        for k in ks:
            arm, n_ins = insert_vz(tq[i], math.radians(k * theta0_deg), link)
            if k != 0:
                # GVZ FIX (review, 25 Aug): n_ins counted MATCHING GATES, not
                # inserted rz, so "n_ins == n_link" compared a number with
                # itself and passed even when nothing was inserted (n_link = 0
                # from a bad --link, or theta0 = 0). The rz delta is now the
                # test, and n_link = 0 stops the run.
                assert n_ins == n_link, "GVZ FAIL: insertion count"
                ca = Counter(dict(tq[i].count_ops()))
                cb = Counter(dict(arm.count_ops()))
                assert cb["rz"] - ca["rz"] == n_link, \
                    "GVZ FAIL: %d rz inserted, expected %d" % (cb["rz"] - ca["rz"], n_link)
                for op in set(ca) | set(cb):
                    if op == "rz":
                        continue
                    assert ca[op] == cb[op], "GVZ FAIL: op %r" % op
            arms[k] = arm
        pubs += [arms[k] for k in ks]
        pubkeys += [[k] + keys[i] for k in ks]
    assert n_link > 0, ("GVZ FAIL: link %s carries no two-qubit gate in the "
                        "compiled round; the arms would be identical and the "
                        "job would measure nothing" % (tuple(sorted(link)),))
    assert theta0_deg != 0.0, "GVZ FAIL: theta0 = 0 inserts nothing"
    # equal/opposite pairs by construction (same insertion sites, k vs -k)
    tgt = backend.target
    assert all(tgt["rz"][(q,)].duration == 0 for q in S.TRIPLE), \
        "GVZ FAIL: rz not virtual"
    return pubs, pubkeys, n_link


def main_submit(args):
    link = frozenset(int(x) for x in args.link.split(","))
    ks = tuple(int(x) for x in args.ks.split(","))
    backend = S.service().backend("ibm_kingston")
    pubs, keys, n_link = build_arms(backend, args.theta0, link, ks)
    print("GVZ PASS: %d arms x 72 settings on link %s; %d rz insertions per "
          "corrected arm circuit; rz virtual on all of %s"
          % (len(ks), tuple(sorted(link)), n_link, S.TRIPLE))
    c = cost(len(pubs), args.shots)
    jobfile = os.path.join(HERE, "rung11b_job%s.json"
                           % ("" if args.tag == "main" else "_" + args.tag))
    print("job: %d circuits x %d shots = ~%.0f s QPU (price from %s)"
          % (len(pubs), args.shots, c, W.source()))
    u = S.service().usage()
    used = u["usage_consumed_seconds"]
    print("safety-loaded (x%.2f): %.0f s" % (SAFETY, c * SAFETY))
    print("live usage: %ss consumed; projected after run %ss (cap %s)"
          % (used, int(used + c * SAFETY), CAP_TOTAL))
    print("output ->", os.path.basename(jobfile))
    if used + c * SAFETY > CAP_TOTAL:
        print("BUDGET GUARD: would pass cap; refusing."); return
    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit (window 3)."); return
    if os.path.exists(jobfile):
        print("REFUSING: %s already exists. A submitted job record is never "
              "overwritten (that is how a control silently replaces the main "
              "run). Move it aside or pass a different --tag."
              % os.path.basename(jobfile)); return
    job = SamplerV2(mode=backend).run(pubs, shots=args.shots)
    json.dump(dict(job_id=job.job_id(), backend=backend.name,
                   layout=list(S.TRIPLE), shots=args.shots, keys=keys,
                   theta0_deg=args.theta0, ks=list(ks), n_link=n_link,
                   link=sorted(link), tag=args.tag,
                   est_cost_s=c * SAFETY, est_cost_unloaded_s=c,
                   cost_source=W.source(),
                   plan="window3_plan_v2.txt (Amendment 1, 25 Aug 2026)",
                   predictions=["V1 V of significant depth, minimum displaced",
                                "V2 minimum below plain by >=3 sigma, direction "
                                "recorded as an OUTPUT (sign clause withdrawn, A10)",
                                "V2P side-prediction: the minimum lies at k = +1",
                                "V3 eta falls, eta-prime flat",
                                "V4 measured and isotropic integers agree",
                                "ASYM ang(+k)-ang(-k) != 0, two-sided (A8)"]),
              open(jobfile, "w"), indent=1)
    print("\nSUBMITTED. job id:", job.job_id())


def main_analyze():
    """Window-3 amendment (25 Aug 2026): estimators and the G-VZ verdict now
    come from w3_vfit (hyperbola primary, bootstrap sigma, signed-arm
    asymmetry, guarded theta write). See that module's header for why."""
    import w3_vfit as V
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"
    meta = json.load(open(os.path.join(
        HERE, "rung11b_job%s.json" % ("" if tag == "main" else "_" + tag))))
    job = S.service().job(meta["job_id"])
    print("job %s: %s" % (meta["job_id"], job.status()))
    res = job.result()
    arms = {k: {} for k in meta["ks"]}
    for i, key in enumerate(meta["keys"]):
        k, prep, mea, f, l = key
        counts = res[i].data.c.get_counts()
        arms[k][(prep, mea, f, l)] = counts.get("0", 0) / meta["shots"]
    rows = []
    for k in sorted(arms, key=lambda k: k * meta["theta0_deg"]):
        b = V.bundle_with_sigma(arms[k], meta["shots"])
        b["k"] = k
        b["applied_deg"] = k * meta["theta0_deg"]
        rows.append(b)
        print("k=%+d (%+.2f deg): angle=%5.2f +/-%.2f  eta=%.4f +/-%.4f  "
              "eta'=%.4f  idx=%d model=%d"
              % (k, b["applied_deg"], b["angle_deg"], b["sd_angle"],
                 b["eta"], b["sd_eta"], b["eta_after_rot"],
                 b["index_measured"], b["index_isotropic"]))

    v = V.gvz_verdict(rows, meta["theta0_deg"])
    print("\n--- GATE G-VZ ---")
    print("  shot-noise floor        : %.2f deg / arm" % v["sigma_bar_deg"])
    if v["hyperbola"]:
        h = v["hyperbola"]
        print("  hyperbola (PRIMARY)     : theta_opt %+.2f deg, floor %.2f deg, "
              "rms resid %.2f deg" % (h["theta_opt"], h["floor_deg"], h["rms_resid"]))
    if v["parabola3"]:
        print("  parabola on k=0,-1,-2   : theta_opt %s"
              % ("%+.2f deg" % v["parabola3"]["theta_opt"]
                 if v["parabola3"]["theta_opt"] is not None else "not convex"))
    if v["parabola5"]:
        print("  parabola on 5 arms (diag): theta_opt %s   [known +0.2 deg bias]"
              % ("%+.2f deg" % v["parabola5"]["theta_opt"]
                 if v["parabola5"]["theta_opt"] is not None else "not convex"))
    for nm in ("asym1", "asym2"):
        a = v[nm]
        if a:
            print("  %s ang(+k)-ang(-k)    : %+.2f +/- %.2f deg  = %+.1f sigma"
                  % (nm.upper(), a["diff_deg"], a["sd"], a["nsigma"]))
    print("  minimum below plain     : %.2f deg = %.1f sigma (arm k=%+d)"
          % (v["drop_deg"] or 0, v["drop_nsigma"] or 0, v["best_k"]))
    print("  V1 %s   V2 %s   V3 %s   V4 %s   ASYM %s"
          % tuple("PASS" if v[x] else "fail"
                  for x in ("V1", "V2", "V3", "V4", "ASYM_signed")))
    if v["both_signed_same_direction"]:
        print("  *** both signed arms moved the same way: DRIFT ARTEFACT ***")
    print("  PATH: %s" % v["path"])

    out = dict(job_id=meta["job_id"], shots=meta["shots"],
               theta0_deg=meta["theta0_deg"], rows=rows, verdict=v,
               p0={str(k): {str(kk): vv for kk, vv in arms[k].items()}
                   for k in arms})
    rf = os.path.join(HERE, "rung11b_result%s.json"
                      % ("" if tag == "main" else "_" + tag))
    json.dump(out, open(rf, "w"), indent=1, default=str)
    print("\nwrote", os.path.basename(rf))
    # CONTROL FIX (review, 25 Aug): keying this on the filename tag meant one
    # forgotten --tag wrote w3_theta.json from a wrong-link control. The link
    # recorded in the job file is the authority.
    # Amendment 1b (25 Aug, post W3-1): the guard keys on the LINK alone. The
    # tag must not gate it, because risk R5 fired and the re-centred repeat is
    # a tagged run on the carrier link that MUST be allowed to write theta.
    link_used = tuple(meta.get("link", [21, 36]))
    if link_used != (21, 36):
        print("control run (link %s): w3_theta.json is never written from a "
              "control." % (link_used,))
        return
    if v["write_theta"]:
        json.dump(dict(theta_opt_deg=v["theta_opt_deg"], source=meta["job_id"],
                       estimator="hyperbola", gate="G-VZ passed",
                       half_span_deg=v["half_span_deg"]),
                  open(os.path.join(HERE, "w3_theta.json"), "w"), indent=1)
        print("wrote w3_theta.json: theta_opt %+.2f deg -> rung8 corrected arms"
              % v["theta_opt_deg"])
    else:
        why = ("outside scanned span" if not v["theta_opt_in_span"]
               else "drift artefact" if v["both_signed_same_direction"]
               else "gate not passed")
        print("w3_theta.json NOT written (%s); rung8 stays plain-only." % why)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        main_analyze()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--yes", action="store_true")
        ap.add_argument("--shots", type=int, default=SHOTS)
        ap.add_argument("--theta0", type=float, default=THETA0_DEG)
        ap.add_argument("--link", type=str, default="21,36",
                        help="link carrying the insertions (control: 21,22)")
        ap.add_argument("--ks", type=str, default="0,-1,1,-2,2")
        ap.add_argument("--tag", type=str, default="main",
                        help="'main' or e.g. 'control2122'")
        ap.parse_args(namespace=(ns := argparse.Namespace()))
        main_submit(ns)
