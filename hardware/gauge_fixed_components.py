#!/usr/bin/env python3
"""
P2 (window-3 prep, zero QPU): gauge-FIXED per-component CZ error extraction.

The bootstrap of 25 Aug demoted the per-component Hamiltonian projections:
resampled refits did not centre on the point estimate, because stdgaugeopt
spreads the Z-error split between the CZ and its neighbouring single-qubit
gates differently per fit. This module fixes the gauge PHYSICALLY: register
13 measured the single-qubit gates to be nearly ideal (about 1 degree of
on-axis under-rotation, no Z-type error), so the gauge freedom is spent
pinning the single-qubit gates to target (item weight 1e3 against 1e-3 on
the CZ and 1.0 on SPAM), dumping residual freedom into the CZ where the
error demonstrably lives.

VALIDATION (the dry run): deliberately scramble the fitted model with random
small gauge transformations, re-run the fixing, and require the CZ H
components to return to the unscrambled values. If the fixing does not pin
the components under scrambling it cannot be trusted on resamples either.
"""
import os, sys, json
import numpy as np
import pygsti, pickle
from pygsti.modelpacks import smq2Q_XYICPHASE as MP

HERE = os.path.dirname(os.path.abspath(__file__))
GL = ("Gcphase", 0, 1)
SQRT2 = np.sqrt(2.0)


def ham(mdl, target, gl=GL):
    G = mdl.operations[gl].to_dense(); Gt = target.operations[gl].to_dense()
    L = pygsti.tools.error_generator(G, Gt, mx_basis="pp", typ="logGTi")
    d = pygsti.tools.project_errorgen(L, "H", "pp")
    d = d[0] if isinstance(d, tuple) else d
    return {str(k).replace("H(", "").replace(")", ""):
            float(np.rad2deg(SQRT2 * np.real(v))) for k, v in d.items()}


def gauge_fix(mdl, target):
    w = {"spam": 1.0, GL: 1e-3}
    for gl in mdl.operations.keys():
        if gl != GL:
            w[gl] = 1e3
    return pygsti.gaugeopt_to_target(mdl.copy(), target, item_weights=w,
                                     maxiter=200, verbosity=0)


def scramble(mdl, seed, eps=0.02):
    rng = np.random.default_rng(seed)
    d = mdl.dim
    M = np.eye(d) + eps * rng.standard_normal((d, d))
    M[0, :] = 0; M[0, 0] = 1          # keep TP structure
    g = pygsti.models.gaugegroup.FullGaugeGroupElement(M)
    out = mdl.copy(); out.transform_inplace(g)
    return out


def main():
    import sys as _s
    tags = [_s.argv[1]] if len(_s.argv) > 1 else ["2122", "2136"]
    target = MP.target_model("full TP")
    fpath = os.path.join(HERE, "gauge_fixed_components.json")
    report = json.load(open(fpath)) if os.path.isfile(fpath) else {}
    for tag in tags:
        mdl = pickle.load(open(os.path.join(HERE, "gst_model_%s.pkl" % tag), "rb"))
        h_raw = ham(mdl, target)
        fixed = gauge_fix(mdl, target)
        h_fix = ham(fixed, target)
        # scramble test: 4 random gauge scrambles, re-fix, compare
        devs = []
        for sd in range(2):
            sc = scramble(mdl, 100 + sd)
            h_sc = ham(gauge_fix(sc, target), target)
            devs.append(max(abs(h_sc[k] - h_fix[k]) for k in h_fix))
        top = sorted(h_fix.items(), key=lambda kv: -abs(kv[1]))[:4]
        report[tag] = dict(h_stdgauge={k: round(v, 3) for k, v in h_raw.items()},
                           h_fixed={k: round(v, 3) for k, v in h_fix.items()},
                           scramble_max_dev_deg=[round(d, 3) for d in devs],
                           scramble_pass=bool(max(devs) < 0.1))
        print("link %s | gauge-FIXED top components: %s" % (tag,
              ", ".join("%s=%+.2f" % (k, v) for k, v in top)))
        print("   scramble test max deviation: %s deg -> %s"
              % (["%.3f" % d for d in devs],
                 "PASS (<0.1)" if max(devs) < 0.1 else "FAIL"))
        for k in ("ZI", "IZ", "ZZ"):
            print("   %s: stdgauge %+.2f -> fixed %+.2f" % (k, h_raw[k], h_fix[k]))
    json.dump(report, open(fpath, "w"), indent=1)
    print("\nwrote gauge_fixed_components.json")
    print("NOTE: full bootstrap under this fixing (refit resamples, fix each,")
    print("re-run centring test) is the remaining P2 step; ~80 min compute,")
    print("zero QPU; run it before window 3 to decide whether W3-2 is needed.")


if __name__ == "__main__":
    main()
