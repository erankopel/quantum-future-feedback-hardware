#!/usr/bin/env python3
"""
Rung 9 analysis: 1-qubit GST on the message qubit. Explorer.

Fetches the GST job, rebuilds the pyGSTi DataSet in circuit order, runs
StandardGST (full TP), and extracts the two things the manuscript wants:
  * SPAM, separated: the estimated prep and POVM error (so we can quote how
    much of eta is state-prep/measurement rather than the round).
  * the gate-associated coherent rotation: for each native gate the
    Hamiltonian error-generator projections, converted to a coherent angle by
    angle_rad = sqrt(2) * H_coeff  (validated on a planted 3.00 deg rotation).
Writes rung9_result.json.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import submit_core as S
import pygsti
from pygsti.modelpacks import smq1Q_XY

SQRT2 = np.sqrt(2.0)


def coherent_angles_deg(mdl, target, gl):
    G = mdl.operations[gl].to_dense()
    Gt = target.operations[gl].to_dense()
    L = pygsti.tools.error_generator(G, Gt, mx_basis="pp", typ="logGTi")
    d = pygsti.tools.project_errorgen(L, "H", "pp")
    d = d[0] if isinstance(d, tuple) else d
    ang = {str(k): float(np.rad2deg(SQRT2 * np.real(v))) for k, v in d.items()}
    total = float(np.rad2deg(SQRT2 * np.sqrt(sum(np.real(v) ** 2
                                                 for v in d.values()))))
    return ang, total


def stochastic_rates(mdl, target, gl):
    G = mdl.operations[gl].to_dense()
    Gt = target.operations[gl].to_dense()
    L = pygsti.tools.error_generator(G, Gt, mx_basis="pp", typ="logGTi")
    d = pygsti.tools.project_errorgen(L, "S", "pp")
    d = d[0] if isinstance(d, tuple) else d
    return {str(k): float(np.real(v)) for k, v in d.items()}


def main():
    meta = json.load(open(os.path.join(HERE, "rung9_job.json")))
    service = S.service()
    job = service.job(meta["job_id"])
    print("job %s: %s" % (meta["job_id"], job.status()))
    res = job.result()

    edesign = smq1Q_XY.create_gst_experiment_design(meta["maxL"])
    circs = list(edesign.all_circuits_needing_data)
    assert len(circs) == meta["n_circuits"], "circuit count mismatch"
    # gate: stored strings must match the regenerated design order
    assert [c.str for c in circs] == meta["circuit_strs"], "GST order mismatch"

    ds = pygsti.data.DataSet(outcome_labels=[("0",), ("1",)])
    shots = meta["shots"]
    for i, c in enumerate(circs):
        counts = res[i].data.c.get_counts()
        n0 = counts.get("0", 0)
        n1 = counts.get("1", shots - n0)
        ds.add_count_dict(c, {("0",): n0, ("1",): n1})
    ds.done_adding_data()

    data = pygsti.protocols.ProtocolData(edesign, ds)
    print("running StandardGST (full TP) on %d circuits..." % len(circs))
    results = pygsti.protocols.StandardGST(modes=["full TP"],
                                           verbosity=1).run(data)
    est = results.estimates["full TP"]
    mdl = est.models["stdgaugeopt"]
    target = smq1Q_XY.target_model("full TP")

    # ---- SPAM error, separated
    def spam_report():
        rho = mdl.preps["rho0"].to_dense()
        rho_t = target.preps["rho0"].to_dense()
        prep_err = float(np.linalg.norm(rho - rho_t))
        # POVM effect for '0'
        E0 = mdl.povms["Mdefault"]["0"].to_dense()
        E0t = target.povms["Mdefault"]["0"].to_dense()
        meas_err = float(np.linalg.norm(E0 - E0t))
        return dict(prep_l2=prep_err, meas_l2=meas_err)

    spam = spam_report()

    gates = {}
    for gl in mdl.operations.keys():
        ang, total = coherent_angles_deg(mdl, target, gl)
        stoch = stochastic_rates(mdl, target, gl)
        gates[str(gl)] = dict(coherent_angles_deg=ang,
                              coherent_total_deg=total,
                              stochastic=stoch)
        print("  %s: coherent %s deg (|H|=%.2f)  "
              % (gl, {k: round(v, 2) for k, v in ang.items()}, total))

    print("SPAM: prep L2=%.4f  meas L2=%.4f" % (spam["prep_l2"], spam["meas_l2"]))

    out = dict(job_id=meta["job_id"], backend=meta["backend"],
               qubit=meta["qubit"], maxL=meta["maxL"], shots=shots,
               spam=spam, gates=gates)
    json.dump(out, open(os.path.join(HERE, "rung9_result.json"), "w"),
              indent=1, default=str)
    print("\nwrote rung9_result.json")


if __name__ == "__main__":
    main()
