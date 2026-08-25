#!/usr/bin/env python3
"""
Gauge-invariant coherent-rotation angles for the CZ, both links. ZERO QPU.

WHY. The per-component Hamiltonian error-generator projections H(ZI), H(IZ)
reported earlier are GAUGE-DEPENDENT: how a Z-type error is split between the
CZ and the neighbouring single-qubit gates depends on the gauge that
stdgaugeopt happens to land in. A parametric bootstrap exposed this directly --
the resampled fits did NOT centre on the point estimate (H(ZI) point -2.843,
bootstrap mean -1.751), which is the signature of refit non-identifiability
rather than clean shot noise. Per-component angles are therefore not safe to
quote in a manuscript.

WHAT IS SAFE. Gauge transformations act on the error map by similarity,
N -> M N M^{-1}, which preserves its EIGENVALUES. So the eigenphases of
N = G G_target^{-1} are gauge-invariant. For a perfect gate N = I and every
eigenvalue is 1. A coherent (unitary) error puts eigenvalues on the unit
circle at nonzero phase; decoherence pulls |lambda| below 1. The largest
eigenphase is a gauge-invariant measure of the gate's coherent rotation, and
it is the honest quantity to report.

Saves the fitted models so no refit is needed again.
"""
import os, sys, json, pickle
import numpy as np
import pygsti
from pygsti.modelpacks import smq2Q_XYICPHASE as MP
import submit_core as S

HERE = os.path.dirname(os.path.abspath(__file__))


def load_ds(meta, circs):
    svc = S.service()
    ca = [None] * len(circs)
    for j in meta["jobs"]:
        res = svc.job(j["job_id"]).result()
        for i in range(j["n"]):
            ca[j["start"] + i] = res[i].data.c.get_counts()
    ds = pygsti.data.DataSet(outcome_labels=[("00",), ("01",), ("10",), ("11",)])
    for c, counts in zip(circs, ca):
        d = {}
        for k, v in counts.items():
            r = k.replace(" ", "")[::-1]
            d[(r,)] = d.get((r,), 0) + v
        ds.add_count_dict(c, d)
    ds.done_adding_data()
    return ds


def eigenphases(mdl, target, gl):
    G = mdl.operations[gl].to_dense()
    Gt = target.operations[gl].to_dense()
    N = G @ np.linalg.inv(Gt)
    ev = np.linalg.eigvals(N)
    ph = np.degrees(np.angle(ev))
    mag = np.abs(ev)
    order = np.argsort(-np.abs(ph))
    return ph[order], mag[order]


def main():
    out = {}
    target = MP.target_model("full TP")
    gl = ("Gcphase", 0, 1)
    for rec, tag in (("rung10_job_0.json", "2122"),
                     ("rung10_job_2136_0.json", "2136")):
        meta = json.load(open(os.path.join(HERE, rec)))
        ed = MP.create_gst_experiment_design(meta["maxL"], fpr=True)
        circs = list(ed.all_circuits_needing_data)
        mpath = os.path.join(HERE, "gst_model_%s.pkl" % tag)
        if os.path.isfile(mpath):
            mdl = pickle.load(open(mpath, "rb"))
            print("loaded cached model %s" % tag, flush=True)
        else:
            ds = load_ds(meta, circs)
            r = pygsti.protocols.StandardGST(modes=["full TP"], verbosity=0).run(
                pygsti.protocols.ProtocolData(ed, ds))
            mdl = r.estimates["full TP"].models["stdgaugeopt"]
            pickle.dump(mdl, open(mpath, "wb"))
            print("fitted and cached model %s" % tag, flush=True)
        ph, mag = eigenphases(mdl, target, gl)
        out[tag] = {"pair": meta["pair"], "shots": meta["shots"],
                    "eigenphases_deg": ph.tolist(),
                    "eigenmagnitudes": mag.tolist(),
                    "max_abs_phase_deg": float(np.abs(ph).max())}
        print("\nlink %s (pair %s, %d shots): CZ error-map eigenvalues"
              % (tag, meta["pair"], meta["shots"]))
        print("  largest |phase| = %.2f deg  (GAUGE-INVARIANT)"
              % np.abs(ph).max())
        print("  top phases (deg): %s"
              % ", ".join("%+.2f" % x for x in ph[:6]))
        print("  |lambda| of those: %s"
              % ", ".join("%.3f" % x for x in mag[:6]))
    json.dump(out, open(os.path.join(HERE, "gst_gaugeinv.json"), "w"), indent=1)
    print("\nwrote gst_gaugeinv.json")


if __name__ == "__main__":
    main()
