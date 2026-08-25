#!/usr/bin/env python3
"""
W3-0b  COST-MODEL PING.  Amendment of 25 Aug 2026.

WHY THIS EXISTS.  The window-3 ledger is priced with a model fitted on charges
raised against two EARLIER accounts.  The window has opened on a THIRD account
(fresh 600 s), and the pre-registered model under-predicts the 19 historical
charges by up to 5.7 per cent.  Committing 100 s to W3-1 on an unvalidated
price is avoidable: this ping costs ~3.5 s (0.6 per cent of the window) and
recalibrates the model on THIS instance before anything expensive runs.

It is not a throwaway.  The 12 circuits are the z+/z- prep sextet of the real
tomography design, so a successful ping is also a live confirmation that the
round executes on the locked triple and that the message qubit responds.

Usage:  python w3_0b_costping.py           # dry run
        python w3_0b_costping.py --yes     # submit (~3.5 s)
        python w3_0b_costping.py refit     # fetch charge, refit, write model
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2
import submit_core as S

HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "w3_costping.json")
SHOTS = 1024
import w3_cost as W
A_PC, B_PS = W.A_DEF, W.B_DEF
SAFETY = W.SAFETY
CAP_TOTAL = 550                        # same G-STAB cap as every other item


def settings():
    for prep in ("z+", "z-"):
        for meas in ("x", "y", "z"):
            for f, l in ((0, 0), (1, 1)):
                yield prep, meas, f, l


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    backend = S.service().backend("ibm_kingston")
    circs = [S.build_tomo(*st) for st in settings()]
    keys = [list(st) for st in settings()]
    tq = transpile(circs, backend=backend, optimization_level=3,
                   initial_layout=list(S.TRIPLE), seed_transpiler=S.SEED)
    c = len(tq) * (A_PC + B_PS * SHOTS)
    u = S.service().usage()
    print("W3-0b ping: %d circuits x %d shots" % (len(tq), SHOTS))
    print("  predicted %.2f s (safety-loaded %.2f s)" % (c, c * SAFETY))
    print("  live usage: %s s consumed of %s s"
          % (u["usage_consumed_seconds"], u["usage_limit_seconds"]))
    if u["usage_consumed_seconds"] + c * SAFETY > CAP_TOTAL:
        print("BUDGET GUARD: would pass cap %s s; refusing." % CAP_TOTAL); return
    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to submit."); return
    job = SamplerV2(mode=backend).run(tq, shots=SHOTS)
    json.dump(dict(job_id=job.job_id(), n=len(tq), shots=SHOTS, keys=keys,
                   predicted_s=c,
                   usage_before=u["usage_consumed_seconds"]),
              open(META, "w"), indent=1)
    print("SUBMITTED", job.job_id())


def refit():
    m = json.load(open(META))
    svc = S.service(); job = svc.job(m["job_id"])
    print("status:", job.status())
    charged = None
    try:
        mm = job.metrics()
        charged = (mm.get("usage", {}) or {}).get("quantum_seconds") or mm.get("usage_seconds")
    except Exception as e:
        print("metrics err:", str(e)[:120])
    if charged is None:
        charged = svc.usage()["usage_consumed_seconds"] - m["usage_before"]
    pred = m["predicted_s"]
    scale = charged / pred if pred else None
    print("  predicted %.2f s, charged %s s, scale %.3f" % (pred, charged, scale))
    res = job.result()
    ok = sum(1 for i in range(len(m["keys"]))
             if sum(res[i].data.c.get_counts().values()) == m["shots"])
    print("  %d/%d circuits returned full shot counts" % (ok, len(m["keys"])))
    out = dict(a_per_circuit=A_PC * scale, b_per_shot=B_PS * scale,
               scale_vs_refit=scale, charged_s=charged, predicted_s=pred,
               source="W3-0b ping on instance 3, %s" % m["job_id"])
    json.dump(out, open(os.path.join(HERE, "cost_model_w3.json"), "w"), indent=1)
    print("  wrote cost_model_w3.json ->", json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refit":
        refit()
    else:
        main()
