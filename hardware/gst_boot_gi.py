#!/usr/bin/env python3
"""
Bootstrap the GAUGE-INVARIANT CZ eigenphases. ZERO QPU. Checkpoints every
sample so a container restart loses at most one fit.

Usage: python gst_boot_gi.py <tag 2122|2136> <n_new_samples>
"""
import os, sys, json, pickle, time
import numpy as np
import pygsti
from pygsti.modelpacks import smq2Q_XYICPHASE as MP

HERE = os.path.dirname(os.path.abspath(__file__))
GL = ("Gcphase", 0, 1)
REC = {"2122": "rung10_job_0.json", "2136": "rung10_job_2136_0.json"}


def eig(mdl, target):
    N = mdl.operations[GL].to_dense() @ np.linalg.inv(target.operations[GL].to_dense())
    ev = np.linalg.eigvals(N)
    ph = np.degrees(np.angle(ev))
    return float(np.abs(ph).max())


def ham_zi(mdl, target):
    G = mdl.operations[GL].to_dense(); Gt = target.operations[GL].to_dense()
    L = pygsti.tools.error_generator(G, Gt, mx_basis="pp", typ="logGTi")
    d = pygsti.tools.project_errorgen(L, "H", "pp")
    d = d[0] if isinstance(d, tuple) else d
    r = {str(k).replace("H(", "").replace(")", ""): float(np.rad2deg(np.sqrt(2) * np.real(v)))
         for k, v in d.items()}
    return r.get("ZI"), r.get("IZ")


def main():
    tag = sys.argv[1]; n_new = int(sys.argv[2])
    meta = json.load(open(os.path.join(HERE, REC[tag])))
    ed = MP.create_gst_experiment_design(meta["maxL"], fpr=True)
    circs = list(ed.all_circuits_needing_data)
    target = MP.target_model("full TP")
    mdl0 = pickle.load(open(os.path.join(HERE, "gst_model_%s.pkl" % tag), "rb"))
    ck = os.path.join(HERE, "gst_bootgi_%s.json" % tag)
    st = json.load(open(ck)) if os.path.isfile(ck) else {
        "tag": tag, "pair": meta["pair"], "shots": meta["shots"],
        "point_eigphase": eig(mdl0, target),
        "point_ham": list(ham_zi(mdl0, target)), "boots": []}
    print("tag %s  point eigenphase %.2f deg  have %d samples"
          % (tag, st["point_eigphase"], len(st["boots"])), flush=True)
    for b in range(n_new):
        seed = 2000 + len(st["boots"])
        t0 = time.time()
        ds_b = pygsti.data.simulate_data(mdl0, circs, num_samples=meta["shots"],
                                         sample_error="multinomial", seed=seed)
        try:
            m = pygsti.protocols.StandardGST(modes=["full TP"], verbosity=0).run(
                pygsti.protocols.ProtocolData(ed, ds_b)
            ).estimates["full TP"].models["stdgaugeopt"]
            e = eig(m, target); hz, hi = ham_zi(m, target)
            st["boots"].append({"seed": seed, "eigphase": e, "ZI": hz, "IZ": hi})
            json.dump(st, open(ck, "w"), indent=1)
            print("  boot %d  %.0fs  eigphase=%.2f  ZI=%+.2f IZ=%+.2f"
                  % (len(st["boots"]), time.time() - t0, e, hz, hi), flush=True)
        except Exception as ex:
            print("  boot FAILED: %s" % str(ex)[:90], flush=True)
    e = [x["eigphase"] for x in st["boots"]]
    z = [x["ZI"] for x in st["boots"]]
    if len(e) > 1:
        print("\nGAUGE-INVARIANT eigenphase: point %.2f | boot mean %.2f sd %.2f"
              % (st["point_eigphase"], np.mean(e), np.std(e, ddof=1)))
        print("GAUGE-DEPENDENT H(ZI):      point %.2f | boot mean %.2f sd %.2f"
              % (st["point_ham"][0], np.mean(z), np.std(z, ddof=1)))
        print("  centring bias (|point-mean|/sd): eigphase %.2f  vs  H(ZI) %.2f"
              % (abs(st["point_eigphase"] - np.mean(e)) / np.std(e, ddof=1),
                 abs(st["point_ham"][0] - np.mean(z)) / np.std(z, ddof=1)))


if __name__ == "__main__":
    main()
