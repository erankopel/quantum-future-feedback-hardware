#!/usr/bin/env python3
"""
Rung 8 (register 12): the multi-epoch stability study. Window-3 upgrade:
epochs may carry a second, virtual-Z-corrected arm (theta from w3_theta.json,
written by rung11b analyze), alternating by epoch parity, so the series
measures not only the drift of eta and the rotation but whether theta_opt
itself is epoch-stable. Budget guard: refuses to submit past CAP_TOTAL.

The paper documented a same-day drift of +0.030 in eta across four jobs that
were never designed as a stability series, and concluded "the implemented
index is a property of a calibration epoch." This register turns that
anecdote into a measured curve: the paper's baseline tomography (build_tomo,
identical to rung2), re-run as N time-separated epochs on the LOCKED triple
(21,22,36), each recording the full explorer bundle (eta, implemented index,
model index, polar rotation angle, eta-after-rotation, anisotropy, singular
values) PLUS the live calibration snapshot (readout, T1, T2, links,
last_update_date) so drift can be read against recalibration events.

Because billing is by execution seconds, spreading epochs over hours is free;
only the shots cost. Each epoch is 72 circuits x SHOTS. The driver in this
session submits one epoch per cadence step, fetches the previous when it has
run, and stops when the epoch count or the QPU-second cap is reached.

Modes:
    python rung8_epoch.py plan             # DRY: build+gate+price, no QPU
    python rung8_epoch.py submit [arm] [shots]   # e.g. "submit both 512"
    python rung8_epoch.py fetch            # fetch+analyze all ran, unfetched
    python rung8_epoch.py status           # print ledger + usage
"""
import os, sys, json, ast, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qiskit import transpile
from qiskit_ibm_runtime import SamplerV2
import submit_core as S
import paper7_core as C

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "rung8_ledger.json")
THETA_FILE = os.path.join(HERE, "w3_theta.json")
# AMENDMENT 1e, 26 Aug 2026. The registered G-STAB cap of 550 s existed to
# stop an IN-PROGRESS series overrunning the hard 600 s and losing a job in
# flight. Window 3 closed at 511 s with the series complete; what remains are
# discrete single epochs, each priced and guarded before submission, so the
# 50 s standoff no longer buys anything. Raised to 590, which still leaves a
# 10 s margin against the hard limit. Logged as a deviation.
CAP_TOTAL = 590
import w3_cost as W
SAFETY = W.SAFETY
SHOTS = 1024          # per MC: se(eta)=0.0031, 10sigma on the documented drift
                      # 21.2 s/epoch plain, 42.3 s/epoch both (corrected model)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_ledger():
    if os.path.isfile(LEDGER):
        return json.load(open(LEDGER))
    return {"plan": {"shots": SHOTS}, "epochs": []}


def save_ledger(led):
    json.dump(led, open(LEDGER, "w"), indent=1, default=str)


def cal_snapshot(backend):
    tgt = backend.target
    props = backend.properties()

    def safe(fn, *a):
        try:
            return fn(*a)
        except Exception:
            return None
    snap = {"qubits": {}}
    for q in S.TRIPLE:
        t1 = safe(props.t1, q); t2 = safe(props.t2, q)
        snap["qubits"][str(q)] = dict(
            readout_err=tgt["measure"][(q,)].error,
            t1_us=(t1 * 1e6 if t1 else None),
            t2_us=(t2 * 1e6 if t2 else None))
    tqn = [n for n in ("ecr", "cz", "cx") if n in tgt][0]
    snap["twoq_gate"] = tqn
    snap["links_on_M"] = {str(tuple(sorted(e))): tgt[tqn][e].error
                          for e in tgt[tqn].keys() if S.TRIPLE[0] in e}
    snap["last_update_date"] = safe(lambda: str(props.last_update_date))
    return snap


def _corrected(tq_list, theta_deg):
    """Insert rz(theta) on the message qubit after every (21,36) CZ, and GATE
    the result. Review fix, 25 Aug: the corrected-arm check used to live only
    in the dry branch, was advisory there, and hardcoded 15; the real submit
    path ran no check at all. It is now enforced here, on the submit path, with
    the insertion count DERIVED from the compiled circuit."""
    import math
    from collections import Counter
    LINK = frozenset((21, 36)); M = S.TRIPLE[0]
    assert theta_deg is not None and theta_deg != 0.0, \
        "corrected arm asked for with theta %r" % (theta_deg,)
    out = []
    for tq in tq_list:
        n_link = sum(1 for inst in tq.data
                     if inst.operation.name in ("cz", "ecr", "cx")
                     and frozenset(tq.find_bit(q).index
                                   for q in inst.qubits) == LINK)
        assert n_link > 0, "GATE FAIL: no (21,36) two-qubit gate in the round"
        c = tq.copy_empty_like()
        for inst in tq.data:
            c.append(inst)
            if inst.operation.name in ("cz", "ecr", "cx"):
                qs = frozenset(tq.find_bit(q).index for q in inst.qubits)
                if qs == LINK:
                    c.rz(math.radians(theta_deg), c.qubits[M])
        ca, cb = Counter(dict(tq.count_ops())), Counter(dict(c.count_ops()))
        assert cb["rz"] - ca["rz"] == n_link, \
            "GATE FAIL: %d rz inserted, expected %d" % (cb["rz"] - ca["rz"], n_link)
        for op in (set(ca) | set(cb)) - {"rz"}:
            assert ca[op] == cb[op], "GATE FAIL: op %r changed" % op
        out.append(c)
    return out


def submit_epoch(shots=SHOTS, arm=None, dry=False):
    """arm: 'plain', 'both', or None = alternate (even epochs 'both' when
    w3_theta.json exists, odd 'plain').  dry=True builds, gate-checks and
    prices the epoch and returns WITHOUT touching the QPU."""
    led = load_ledger()
    k = len(led["epochs"])
    theta = None
    if os.path.isfile(THETA_FILE):
        theta = json.load(open(THETA_FILE)).get("theta_opt_deg")
    if arm is None:
        arm = "both" if (theta is not None and k % 2 == 0) else "plain"
    if arm == "both" and theta is None:
        print("no w3_theta.json; falling back to plain"); arm = "plain"

    backend = S.service().backend("ibm_kingston")
    circuits, keys = [], []
    for (prep, meas, f, l) in S.all_settings():
        circuits.append(S.build_tomo(prep, meas, f, l))
        keys.append([prep, meas, f, l])
    tq = transpile(circuits, backend=backend, optimization_level=3,
                   initial_layout=list(S.TRIPLE), seed_transpiler=S.SEED)
    pubs = list(tq); pubkeys = [[0] + kk for kk in keys]
    if arm == "both":
        pubs += _corrected(tq, theta)
        pubkeys += [[1] + kk for kk in keys]

    cost = W.cost(len(pubs), shots)     # safety-loaded
    u = S.service().usage(); used = u["usage_consumed_seconds"]
    if dry:
        # the corrected-arm gate already ran inside _corrected() above and
        # would have raised; reaching here means it passed.
        print("DRY epoch %d: arm=%s%s, %d circuits x %d shots -> %.1f s "
              "(safety-loaded, price from %s); used %ss of cap %ss; "
              "corrected-arm gate: %s"
              % (k, arm, ("" if theta is None else ", theta=%+.2f" % theta),
                 len(pubs), shots, cost, W.source(), used, CAP_TOTAL,
                 "PASS" if arm == "both" else "n/a"))
        if used + cost > CAP_TOTAL:
            print("  -> BUDGET GUARD would REFUSE this epoch.")
        return None
    if used + cost > CAP_TOTAL:
        print("BUDGET GUARD: %ss used + %.0fs would pass cap %s; refusing."
              % (used, cost, CAP_TOTAL))
        return None
    sampler = SamplerV2(mode=backend)
    job = sampler.run(pubs, shots=shots)
    led["epochs"].append(dict(epoch=k, job_id=job.job_id(), shots=shots,
                              submitted_utc=now_utc(), keys=pubkeys,
                              arm=arm, theta_deg=(theta if arm == "both"
                                                  else None),
                              cal_at_submit=cal_snapshot(backend),
                              fetched=False))
    save_ledger(led)
    print("submitted epoch %d (%s%s): job %s (~%.0f s)"
          % (k, arm, ", theta=%.2f" % theta if arm == "both" else "",
             job.job_id(), cost))
    return job.job_id()


def _job_times(job):
    t = {}
    for key in ("Created", "Running", "Completed", "Finished"):
        pass
    try:
        m = job.metrics()
        ts = m.get("timestamps", {}) if isinstance(m, dict) else {}
        t = {k: ts.get(k) for k in ("created", "running", "finished")}
        t["usage_seconds"] = m.get("usage", {}).get("quantum_seconds") \
            if isinstance(m.get("usage"), dict) else m.get("usage_seconds")
    except Exception as e:
        t["metrics_err"] = str(e)[:120]
    return t


def fetch_ready():
    led = load_ledger()
    service = S.service()
    backend = service.backend("ibm_kingston")
    changed = 0
    for ep in led["epochs"]:
        if ep.get("fetched"):
            continue
        job = service.job(ep["job_id"])
        st = str(job.status())
        if st not in ("DONE", "JobStatus.DONE", "Completed"):
            print("epoch %d: %s (not ready)" % (ep["epoch"], st))
            continue
        res = job.result()
        arms = {}
        for i, key in enumerate(ep["keys"]):
            a, prep, mea, f, l = key
            counts = res[i].data.c.get_counts()
            arms.setdefault(a, {})[(prep, mea, f, l)] = \
                counts.get("0", 0) / ep["shots"]
        bundle = C.analyze(arms[0])
        ep.update(bundle)
        if 1 in arms:
            ep["corrected"] = C.analyze(arms[1])
        ep["p0"] = {str(a): {str(kk): v for kk, v in arms[a].items()}
                    for a in arms}
        ep["fetched"] = True
        ep["fetched_utc"] = now_utc()
        ep["job_times"] = _job_times(job)
        ep["cal_at_fetch"] = cal_snapshot(backend)
        changed += 1
        print("epoch %d fetched: eta=%.4f index=%d model=%d angle=%.1f "
              "aniso=%.3f" % (ep["epoch"], bundle["eta"],
                              bundle["index_measured"], bundle["index_isotropic"],
                              bundle["angle_deg"], bundle["anisotropy"]))
    # usage
    try:
        led["usage"] = service.usage()
    except Exception:
        pass
    save_ledger(led)
    return changed


def status():
    led = load_ledger()
    print(json.dumps({"n_epochs": len(led["epochs"]),
                      "usage": led.get("usage", {}).get("usage_consumed_seconds"),
                      "epochs": [{k: e.get(k) for k in
                                  ("epoch", "eta", "index_measured",
                                   "index_isotropic", "angle_deg",
                                   "fetched", "submitted_utc")}
                                 for e in led["epochs"]]}, indent=1, default=str))


# keys aren't stored in submit_epoch above; store them there
def _patch_keys_note():
    pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "submit":
        arm = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ("plain", "both") else None
        sh = int(sys.argv[3]) if len(sys.argv) > 3 else SHOTS
        submit_epoch(shots=sh, arm=arm)
    elif mode in ("plan", "dry"):
        submit_epoch(dry=True)
    elif mode == "fetch":
        fetch_ready()
    else:
        status()
