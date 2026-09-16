#!/usr/bin/env python3
"""
Paper A checks (float64, explorer). Re-derives from the deposited counts of the
first hardware run (rung2_result.json) the numbers Paper A quotes, and adds two
checks the QST manuscript did not report:

  1. complete positivity of the reconstructed one-round map (min eigenvalue of
     its Choi matrix), and the index of its projection to the nearest CPTP map
     if it is not CP;
  2. a SPAM-corrected pair, using the GST preparation and readout estimates of
     rung9_spam.json, and the index of that pair.

Everything is float64 and labelled explorer, exactly as in the series.
"""
import os, sys, json
import numpy as np

# Locate the deposited data and the series' float64 model. The script runs
# either beside copies of the JSONs and velocity_probe.py, or from tools/ in
# the repository (data in ../data, model in ../src).
HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, "..", "src")):
    if os.path.exists(os.path.join(cand, "velocity_probe.py")):
        sys.path.insert(0, cand)
        break
DATA = HERE if os.path.exists(os.path.join(HERE, "rung2_result.json")) \
    else os.path.join(HERE, "..", "data")
from velocity_probe import U, QR1, HOLD, tau, bloch_pair, choi_pt_lmin, SIG, I2

P_HOLD = float(HOLD)
U8 = U(QR1["theta"], QR1["phi"], QR1["kappa"], QR1["beta"])
Aid, cid = bloch_pair(U8, tau(P_HOLD))

PREP = ["z+", "z-", "x+", "x-", "y+", "y-"]


def pair_from_p0(p0):
    """Verbatim linear combinations of the series' register-3 read-off."""
    w = {0: (1 + P_HOLD) / 2, 1: (1 - P_HOLD) / 2}
    exp = {k: 2 * v - 1 for k, v in p0.items()}
    r_out = {}
    for prep in PREP:
        r_out[prep] = np.array([sum(w[f] * w[l] * exp[(prep, ms, f, l)]
                                    for f in (0, 1) for l in (0, 1))
                                for ms in ("x", "y", "z")])
    c = (r_out["z+"] + r_out["z-"]) / 2
    A = np.zeros((3, 3))
    for j, (pp, pm) in enumerate((("x+", "x-"), ("y+", "y-"), ("z+", "z-"))):
        A[:, j] = (r_out[pp] - r_out[pm]) / 2
    return A, c


def fit_eta(A):
    return 1.0 - float(np.sum(A * Aid) / np.sum(Aid * Aid))


def choi(A, c):
    """Choi matrix J = sum_ij |i><j| (x) Phi(|i><j|), normalised to trace 1."""
    J = np.zeros((4, 4), complex)
    for i in range(2):
        for j in range(2):
            E = np.zeros((2, 2), complex); E[i, j] = 1
            t = np.trace(E) / 2
            v = np.array([np.trace(SIG[k] @ E) / 2 for k in range(3)])
            w = t * c + A @ v
            M = t * I2 + sum(w[k] * SIG[k] for k in range(3))
            J += np.kron(E, M) / 2
    return J


def implemented_index(A, c, nmax=400):
    Ac, cc = np.eye(3), np.zeros(3)
    for r in range(1, nmax + 1):
        Ac = A @ Ac
        cc = A @ cc + c
        if choi_pt_lmin(Ac, cc) >= 0:
            return r
    return None


def nearest_cptp_pair(A, c, iters=2000):
    """Project the Choi matrix to the nearest trace-1 PSD matrix with the
    trace-preserving constraint (alternating projections, Frobenius), then read
    the affine pair back off it."""
    # index layout: J[i*2+a, j*2+b] with i, j input and a, b output indices
    J = choi(A, c)
    for _ in range(iters):
        # PSD projection
        w, V = np.linalg.eigh((J + J.conj().T) / 2)
        w = np.clip(w, 0, None)
        J = (V * w) @ V.conj().T
        # trace-preserving projection: Tr_out J = I/2
        Jr = J.reshape(2, 2, 2, 2)
        marg = np.einsum("iaja->ij", Jr)           # trace over the output index
        corr = (marg - I2 / 2) / 2
        Jr = Jr - np.einsum("ij,ab->iajb", corr, I2)
        J = Jr.reshape(4, 4)
    # read the pair back: Phi(E)[a,b] = 2 * sum_{k,i} E[k,i] J[k,a,i,b]
    def Phi(E):
        Jr = J.reshape(2, 2, 2, 2)
        return 2 * np.einsum("ki,kaib->ab", E, Jr)
    cnew = np.array([np.trace(SIG[i] @ Phi(0.5 * I2)).real for i in range(3)])
    Anew = np.zeros((3, 3))
    for j in range(3):
        Ps = Phi(SIG[j])
        for i in range(3):
            Anew[i, j] = (0.5 * np.trace(SIG[i] @ Ps)).real
    return Anew, cnew, J


def main():
    out = {}
    HW = json.load(open(os.path.join(DATA, "rung2_result.json")))
    p0 = {eval(k): v for k, v in HW["p0"].items()}
    A, c = pair_from_p0(p0)
    eta = fit_eta(A)
    sv = sorted(np.linalg.svd(A, compute_uv=False).tolist(), reverse=True)
    n_meas = implemented_index(A, c)
    n_model = implemented_index((1 - eta) * Aid, (1 - eta) * cid)
    out["baseline"] = dict(eta=eta, singular_values=sv, norm_c=float(np.linalg.norm(c)),
                           index_measured=n_meas, index_isotropic_model=n_model,
                           recorded=dict(eta=HW["eta"], index=HW["implemented_index"],
                                         model=HW["model_index"], certified_lower=HW["certified_lower"]))
    assert n_meas == HW["implemented_index"] and n_model == HW["model_index"]
    assert abs(eta - HW["eta"]) < 1e-9

    # ideal pair sanity: noiseless index of QR1 at the hold polarisation
    out["ideal"] = dict(index=implemented_index(Aid, cid),
                        singular_values=sorted(np.linalg.svd(Aid, compute_uv=False).tolist(), reverse=True),
                        norm_c=float(np.linalg.norm(cid)))

    # 1. complete positivity of the reconstructed map
    J = choi(A, c)
    ev = np.linalg.eigvalsh((J + J.conj().T) / 2)
    out["cp_check"] = dict(choi_eigenvalues=ev.tolist(), min_eigenvalue=float(ev[0]),
                           trace=float(np.trace(J).real), is_cp=bool(ev[0] >= -1e-12))
    Ap, cp, Jp = nearest_cptp_pair(A, c)
    evp = np.linalg.eigvalsh((Jp + Jp.conj().T) / 2)
    out["cp_projection"] = dict(min_eigenvalue_after=float(evp[0]),
                                frob_change_A=float(np.linalg.norm(Ap - A)),
                                change_c=float(np.linalg.norm(cp - c)),
                                eta_after=fit_eta(Ap),
                                index_after=implemented_index(Ap, cp))

    # 2. SPAM-corrected pair from the GST estimates (later epoch, same device)
    S = json.load(open(os.path.join(DATA, "rung9_spam.json")))
    p_ro = S["p_ro_avg"]
    r_prep = float(np.linalg.norm(S["prep_bloch"]))
    ro_scale = 1 - 2 * p_ro
    A_s = A / (ro_scale * r_prep)
    c_s = c / ro_scale
    eta_s = fit_eta(A_s)
    n_s = implemented_index(A_s, c_s)
    Js = choi(A_s, c_s)
    evs = np.linalg.eigvalsh((Js + Js.conj().T) / 2)
    out["spam_corrected"] = dict(p_ro_avg=p_ro, prep_bloch_length=r_prep,
                                 readout_scale=ro_scale, total_scale=ro_scale * r_prep,
                                 eta_spam_implied=1 - ro_scale * r_prep,
                                 eta_spam_recorded=S["eta_spam"],
                                 eta_corrected=eta_s, index_corrected=n_s,
                                 index_isotropic_model_corrected=implemented_index((1 - eta_s) * Aid, (1 - eta_s) * cid),
                                 choi_min_eigenvalue=float(evs[0]))
    # readout-only correction (the morning snapshot readout error), prep left alone
    ro_m = 1 - 2 * HW["calibration"]["readout_err_M"]
    A_r, c_r = A / ro_m, c / ro_m
    out["readout_only_corrected"] = dict(readout_err_M=HW["calibration"]["readout_err_M"],
                                         eta=fit_eta(A_r), index=implemented_index(A_r, c_r))

    # sensitivity: how many integers per 0.01 of eta near the measured point
    grid = np.linspace(0.08, 0.13, 51)
    idx = [implemented_index((1 - e) * Aid, (1 - e) * cid) for e in grid]
    out["local_sensitivity"] = dict(eta_grid=grid.tolist(), index=idx)

    # 3. parametric bootstrap of the shot noise: Binomial(shots, p0) per setting
    rng = np.random.default_rng(20260916)
    shots = HW["shots"]
    keys = list(p0.keys())
    pvec = np.array([p0[k] for k in keys])
    B = 4000
    etas, idxs, lmins, anis, etas_s, idxs_s = [], [], [], [], [], []
    for _ in range(B):
        pb = rng.binomial(shots, pvec) / shots
        pb_d = dict(zip(keys, pb))
        Ab, cb = pair_from_p0(pb_d)
        eb = fit_eta(Ab)
        etas.append(eb)
        idxs.append(implemented_index(Ab, cb))
        Jb = choi(Ab, cb)
        lmins.append(float(np.linalg.eigvalsh((Jb + Jb.conj().T) / 2)[0]))
        anis.append(float(np.linalg.norm(Ab - (1 - eb) * Aid) / np.linalg.norm(Aid)))
        As_b, cs_b = Ab / (ro_scale * r_prep), cb / ro_scale
        etas_s.append(fit_eta(As_b))
        idxs_s.append(implemented_index(As_b, cs_b))
    def hist(v):
        u, n = np.unique(np.array(v), return_counts=True)
        return {int(a): float(b / len(v)) for a, b in zip(u, n)}
    out["bootstrap"] = dict(resamples=B, shots=shots,
                            eta_mean=float(np.mean(etas)), eta_sd=float(np.std(etas)),
                            index_distribution=hist(idxs),
                            choi_min_eig_mean=float(np.mean(lmins)), choi_min_eig_sd=float(np.std(lmins)),
                            choi_min_eig_q025=float(np.quantile(lmins, 0.025)),
                            choi_min_eig_q975=float(np.quantile(lmins, 0.975)),
                            anisotropy_sd=float(np.std(anis)), anisotropy_mean=float(np.mean(anis)),
                            spam_corrected_eta_mean=float(np.mean(etas_s)), spam_corrected_eta_sd=float(np.std(etas_s)),
                            spam_corrected_index_distribution=hist(idxs_s))

    json.dump(out, open(os.path.join(HERE, "paperA_checks.json"), "w"), indent=1)
    for k, v in out.items():
        if k == "local_sensitivity":
            steps = [(round(float(g), 4), i) for g, i in zip(grid, idx)]
            print(k, ":", steps[::5])
        else:
            print(k, ":", json.dumps(v, indent=None)[:600])


if __name__ == "__main__":
    main()
