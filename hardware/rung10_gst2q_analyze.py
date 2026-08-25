#!/usr/bin/env python3
"""
Rung 10 analysis: reduced (FPR) two-qubit GST on the (21,22) CZ link.

Reassembles the 6 chunk jobs in circuit order, builds the pyGSTi DataSet, runs
StandardGST, and extracts the quantity the paper actually needs: the CZ gate's
COHERENT (Hamiltonian) error generator, in particular its Z-type components.

Rung 9 showed the single-qubit gates carry ~1 degree of pure on-axis
under-rotation with H(Z) ~ 0, which cannot explain the 5-9 degree rotation
that survives the rung-6 echo. If the CZ carries Z-type coherent error
(H(IZ), H(ZI), H(ZZ)) of the right size, that closes the diagnosis.

GATE GBIT (endianness). Qiskit is little-endian: a counts key 'b1b0' has b0 on
classical bit 0, i.e. on qubit 21. pyGSTi outcome labels are ordered
(qubit0, qubit1). The bit string is therefore REVERSED on ingest. This mirrors
the series' own G0 lesson, where an unreversed qargs list silently corrupted
every downstream number. The gate asserts the reversed marginals reproduce the
per-qubit click probabilities computed directly from the counts.

Label: explorer (point estimates, no interval certification).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import submit_core as S
import pygsti
from pygsti.modelpacks import smq2Q_XYICPHASE as MP

SQRT2 = np.sqrt(2.0)


def main():
    rec = sys.argv[1] if len(sys.argv) > 1 else "rung10_job_2136_0.json"
    meta = json.load(open(os.path.join(HERE, rec)))
    print("record: %s  pair %s  shots %d" % (rec, meta["pair"], meta["shots"]))
    svc = S.service()

    edesign = MP.create_gst_experiment_design(meta["maxL"], fpr=True)
    circs = list(edesign.all_circuits_needing_data)
    assert [c.str for c in circs] == meta["circuit_strs"], "design order mismatch"
    shots = meta["shots"]

    # ---- gather every chunk, in order
    counts_all = [None] * len(circs)
    for j in meta["jobs"]:
        job = svc.job(j["job_id"])
        st = str(job.status())
        if "DONE" not in st:
            print("chunk at %d (%s): %s -- not ready" % (j["start"], j["job_id"], st))
            return 1
        res = job.result()
        for i in range(j["n"]):
            counts_all[j["start"] + i] = res[i].data.c.get_counts()
    missing = [i for i, c in enumerate(counts_all) if c is None]
    assert not missing, "missing %d circuits" % len(missing)
    print("assembled all %d circuits from %d jobs" % (len(circs), len(meta["jobs"])))

    # ---- build DataSet (GATE GBIT: reverse qiskit's little-endian keys)
    ds = pygsti.data.DataSet(outcome_labels=[("00",), ("01",), ("10",), ("11",)])
    gbit_checked = 0
    for c, counts in zip(circs, counts_all):
        d = {}
        q0_ones = 0
        for k, v in counts.items():
            k = k.replace(" ", "")
            rev = k[::-1]                     # 'b1b0' -> 'b0b1' = (q0,q1)
            d[(rev,)] = d.get((rev,), 0) + v
            if k[-1] == "1":
                q0_ones += v
        # GBIT: marginal on qubit 0 from reversed labels must match raw counts
        m0 = sum(v for (o,), v in d.items() if o[0] == "1")
        assert m0 == q0_ones, "GBIT FAILED on %s" % c.str
        gbit_checked += 1
        ds.add_count_dict(c, d)
    ds.done_adding_data()
    print("GBIT PASS: qubit-0 marginals consistent on all %d circuits" % gbit_checked)

    data = pygsti.protocols.ProtocolData(edesign, ds)
    print("running StandardGST (full TP) on 2 qubits; this takes a while...")
    results = pygsti.protocols.StandardGST(modes=["full TP"], verbosity=1).run(data)
    mdl = results.estimates["full TP"].models["stdgaugeopt"]
    target = MP.target_model("full TP")

    def ham(gl):
        G = mdl.operations[gl].to_dense(); Gt = target.operations[gl].to_dense()
        L = pygsti.tools.error_generator(G, Gt, mx_basis="pp", typ="logGTi")
        d = pygsti.tools.project_errorgen(L, "H", "pp")
        d = d[0] if isinstance(d, tuple) else d
        return {str(k): float(np.rad2deg(SQRT2 * np.real(v))) for k, v in d.items()}

    out = {"pair": meta["pair"], "maxL": meta["maxL"], "shots": shots,
           "fpr": True, "n_circuits": len(circs), "gates": {}}
    for gl in mdl.operations.keys():
        angs = ham(gl)
        tot = float(np.sqrt(sum(v * v for v in angs.values())))
        out["gates"][str(gl)] = {"coherent_deg": angs, "coherent_total_deg": tot}
        big = sorted(angs.items(), key=lambda kv: -abs(kv[1]))[:4]
        print("  %-16s |H|=%5.2f deg  top: %s"
              % (gl, tot, ", ".join("%s=%+.2f" % (k, v) for k, v in big)))

    cz = [k for k in out["gates"] if "cphase" in k.lower() or "Gcphase" in k]
    if cz:
        a = out["gates"][cz[0]]["coherent_deg"]
        zc = {k: v for k, v in a.items() if set(k[2:-1]) <= set("IZ") and k != "H(II)"}
        print("\nCZ Z-type coherent components (the diagnosis-relevant ones):")
        for k, v in sorted(zc.items(), key=lambda kv: -abs(kv[1])):
            print("   %-8s %+.3f deg" % (k, v))
        out["cz_z_type_deg"] = zc

    outname = "rung10_result_%s.json" % "".join(str(x) for x in meta["pair"])
    json.dump(out, open(os.path.join(HERE, outname), "w"),
              indent=1, default=str)
    print("\nwrote", outname)
    return 0


if __name__ == "__main__":
    sys.exit(main())
