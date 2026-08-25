#!/usr/bin/env python3
"""
Rung 2, part 2: turn the hardware counts into the paper's three numbers.

Reads rung2_job.json (written by rung2_submit.py), fetches the completed
SamplerV2 job, and produces:

  1. THE POINT ESTIMATE (explorer). The measured affine Bloch pair (A, c),
     the fitted contraction eta against the ideal pair, and the implemented
     index of the measured round-plus-SPAM channel, with the register-2
     curve's prediction alongside.
  2. THE CERTIFICATE (certified-given-shots). The 72-symbol affine
     propagation of register 4 at the actual shot count: a certified bracket
     on the implemented index at joint confidence 1 - 1e-3, Bonferroni over
     the 72 settings.
  3. THE RECORD. rung2_result.json with all of it, plus the calibration
     snapshot carried over from submission.

Honesty notes printed with the output and repeated here: the tomography is
linear inversion, so eta and the index include SPAM; the certificate at a
few thousand shots per setting is weak by design (the paper's exchange
tables say exactly how weak) and the run's value is the point estimate plus
the demonstration that the pipeline closes end to end on real counts.

Usage:  python rung2_analyze.py            # uses rung2_job.json next to it
        python rung2_analyze.py JOB_ID     # override job id
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "anc"))

from qiskit_ibm_runtime import QiskitRuntimeService
import rung1_hardware_bracket as R
import register3_budget as B3
import register4_affine as R4


def main():
    meta = json.load(open(os.path.join(HERE, "rung2_job.json")))
    job_id = sys.argv[1] if len(sys.argv) > 1 else meta["job_id"]
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
               if token else QiskitRuntimeService())
    job = service.job(job_id)
    print("job %s on %s: %s" % (job_id, meta["backend"], job.status()))
    result = job.result()

    shots = meta["shots"]
    p0 = {}
    for i, key in enumerate(meta["keys"]):
        counts = result[i].data.c.get_counts()
        n0 = counts.get("0", 0)
        p0[tuple(key)] = n0 / shots

    # ---- 1. point estimate
    alpha = B3.DELTA / B3.N_SETTINGS
    A, c, _, _ = B3.pair_and_radii(p0, 10**15, alpha)
    Ai, ci = R.bloch_pair(R.U8, R.tau(R.P_HOLD))
    s = float(np.sum(A * Ai) / np.sum(Ai * Ai))
    eta = 1.0 - s
    aniso = float(np.linalg.norm(A - s * Ai) / np.linalg.norm(Ai))
    _, n_impl, _ = R.index_bracket(A, c, 0.0, nmax=300)
    Am, cm = (1 - eta) * Ai, (1 - eta) * ci
    _, n_model, _ = R.index_bracket(Am, cm, 0.0, nmax=300)
    print("\nPOINT ESTIMATE (explorer; includes SPAM)")
    print("  eta = %.5f   anisotropy residual = %.3e" % (eta, aniso))
    print("  implemented index (measured pair) : %s" % n_impl)
    print("  one-parameter model at this eta   : %s   [%s]"
          % (n_model, "consistent" if n_model == n_impl else
             "DEVIATES - report both, the deviation is a result"))

    # ---- 2. certificate at the actual shot count
    Aa, ca = R4.affine_pair(p0, shots, alpha)
    lo, first_ppt, unres = R4.index_bracket_affine(Aa, ca, nmax=300)
    if first_ppt:
        cert = "index in (%d, %d]" % (lo, first_ppt)
    else:
        cert = "index > %d (unresolved from round %s)" % (lo, unres)
    print("\nCERTIFICATE (certified-given-shots; 72-symbol affine, "
          "joint 1 - 1e-3)")
    print("  %s   at %d shots/setting" % (cert, shots))

    json.dump(dict(job_id=job_id, backend=meta["backend"],
                   layout=meta["layout"], shots=shots,
                   calibration=meta["calibration"],
                   eta=eta, anisotropy=aniso,
                   implemented_index=n_impl, model_index=n_model,
                   certified_lower=lo, certified_first_ppt=first_ppt,
                   p0={str(k): v for k, v in p0.items()}),
              open(os.path.join(HERE, "rung2_result.json"), "w"), indent=1)
    print("\nwrote rung2_result.json")


if __name__ == "__main__":
    main()
