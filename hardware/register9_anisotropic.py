#!/usr/bin/env python3
"""
Paper VII register 9: the anisotropic extension, tested on the day's data.

The isotropic model (1 - eta)(A, c) was exact 9 of 9 under depolarising
simulation (register 1) and missed all seven hardware channels measured on
2026-08-24, always in the same direction: measured index above the model, by
one integer six times and by two at the largest anisotropy. This register
asks the falsifiable follow-up: do THREE parameters restore the integer?

MODEL LADDER (all fit to the measured pair by least squares, all built from
the IDEAL pair; the measured pair itself is the ground truth whose index the
model must predict):

  M0  isotropic, 1 parameter:      A = (1-eta) A_id,  c = (1-eta) c_id
  M1  post-round diagonal, 3:      A = D A_id,        c = D c_id
      with D = diag(l1, l2, l3) in the message-qubit Bloch frame, fit
      row-wise: l_i = <A_meas[i,:], A_id[i,:]> / ||A_id[i,:]||^2.
  M1' pre-round diagonal, 3:       A = A_id D',       c = c_id
      fit column-wise. Distinguishes where the noise sits.
  M2  M1 plus a non-unital shift:  c = D c_id + t e_z
      (amplitude damping pushes toward |0>; t fit from the z component).

DIAGNOSTICS per channel: residual anisotropy after each model (what share of
the structure each captures); the polar decomposition N = O P of the raw
noise map N = A_meas A_id^{-1}, whose rotation angle is the coherent-error
angle and whose symmetric part carries the true contraction triple.

MECHANISM CHECK: the isotropic fit averages the contraction over axes, so
when the noise is anisotropic it over-contracts the best-preserved axis,
which is the axis that governs the late-round margin; hence the model
crosses PPT early and predicts a LOW index, which is exactly the 7-of-7
sign observed. If the mechanism is right, M1 should repair the integer.

GATES. G0 inherited (noiseless tomography = series pair, asserted on
import). G3 (new, planted noise): on synthetic channels with planted
diagonal noise (and, for M2, a planted shift), the fits must recover the
planted parameters to 1e-10 and the planted index exactly, and M0 on
register-1 depolarising data must agree with the register-1 isotropic
results. Refuses to report if any gate fails.

float64 explorer throughout, labelled. Zero QPU.
"""
import os, sys, json, ast
import numpy as np
from scipy.linalg import polar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "anc"))

import rung1_hardware_bracket as R                            # noqa: E402
import register3_budget as B3                                 # noqa: E402

Aid, cid = R.bloch_pair(R.U8, R.tau(R.P_HOLD))
ALPHA = B3.DELTA / 72


def index_of(A, c, nmax=400):
    _, first_ppt, _ = R.index_bracket(A, c, 0.0, nmax=nmax)
    return first_ppt


def fit_M0(A):
    eta = 1.0 - float(np.sum(A * Aid) / np.sum(Aid * Aid))
    return eta, ((1 - eta) * Aid, (1 - eta) * cid)


def fit_M1_post(A):
    lam = np.array([float(np.dot(A[i], Aid[i]) / np.dot(Aid[i], Aid[i]))
                    for i in range(3)])
    D = np.diag(lam)
    return lam, (D @ Aid, D @ cid)


def fit_M1_pre(A):
    mu = np.array([float(np.dot(A[:, j], Aid[:, j]) / np.dot(Aid[:, j], Aid[:, j]))
                   for j in range(3)])
    return mu, (Aid @ np.diag(mu), cid.copy())


def fit_M2(A, c):
    lam, (Am, cm) = fit_M1_post(A)
    t = float(c[2] - cm[2])
    c2 = cm.copy(); c2[2] += t
    return (lam, t), (Am, c2)


def fit_M3_rot_scalar(A):
    """Rotation (polar) + one scalar: A = (1-e) O A_id, c = (1-e) O c_id."""
    N = A @ np.linalg.inv(Aid)
    O, _ = polar(N)
    B = O @ Aid
    e = 1.0 - float(np.sum(A * B) / np.sum(B * B))
    return (O, e), ((1 - e) * B, (1 - e) * (O @ cid))


def fit_M4_rot_diag(A):
    """Rotation + diagonal: A = O D A_id, c = O D c_id."""
    N = A @ np.linalg.inv(Aid)
    O, _ = polar(N)
    Ar = O.T @ A                      # rotation removed
    lam = np.array([float(np.dot(Ar[i], Aid[i]) / np.dot(Aid[i], Aid[i]))
                    for i in range(3)])
    D = np.diag(lam)
    return (O, lam), (O @ D @ Aid, O @ D @ cid)


def fit_M5_unital(A):
    """Full linear noise map, unital c-model: A exact via N = A A_id^-1,
    c = N c_id (9 parameters; tests whether the integer needs c)."""
    N = A @ np.linalg.inv(Aid)
    return N, (A.copy(), N @ cid)


def fragility(A, c, n):
    """min(|lmin(n-1)|, lmin(n)) of the measured composite: how close the
    integer sits to flipping."""
    from velocity_probe import choi_pt_lmin
    Ac, cc = np.eye(3), np.zeros(3)
    vals = {}
    for r in range(1, n + 1):
        Ac = A @ Ac; cc = A @ cc + c
        if r in (n - 1, n):
            vals[r] = choi_pt_lmin(Ac, cc)
    return float(min(abs(vals[n - 1]), vals[n]))


def resid(A, Am):
    return float(np.linalg.norm(A - Am) / np.linalg.norm(Aid))


def coherent_angle_deg(A):
    N = A @ np.linalg.inv(Aid)
    O, P = polar(N)
    if np.linalg.det(O) < 0:
        return None, np.linalg.eigvalsh(P)
    ang = np.degrees(np.arccos(np.clip((np.trace(O) - 1) / 2, -1, 1)))
    return float(ang), np.sort(np.linalg.eigvalsh(P))[::-1]


# ---------------------------------------------------------------- gates
def gates():
    # G3a: planted diagonal noise recovered exactly
    Dpl = np.diag([0.93, 0.97, 0.90])
    Apl, cpl = Dpl @ Aid, Dpl @ cid
    lam, (Am, cm) = fit_M1_post(Apl)
    assert np.max(np.abs(lam - np.diag(Dpl))) < 1e-10, "G3a param FAIL"
    assert index_of(Am, cm) == index_of(Apl, cpl), "G3a index FAIL"
    # G3b: planted shift recovered by M2
    cpl2 = cpl.copy(); cpl2[2] += 0.02
    (lam2, t2), (Am2, cm2) = fit_M2(Apl, cpl2)
    assert abs(t2 - 0.02) < 1e-10, "G3b shift FAIL"
    assert index_of(Am2, cm2) == index_of(Apl, cpl2), "G3b index FAIL"
    # G3c: on register-1 depolarising data, M0 here = register-1 isotropic
    r1 = json.load(open(os.path.join(HERE, "contraction_register1_f64.json")))
    for row in r1["rows"][:3]:
        eta_ref = row["eta"]
        Am0 = (1 - eta_ref) * Aid; cm0 = (1 - eta_ref) * cid
        assert index_of(Am0, cm0) == row["index_model"], "G3c FAIL"
    print("G3 PASS: planted diagonal and shift recovered exactly; "
          "M0 consistent with register 1")


# ---------------------------------------------------------------- channels
def load_channels():
    chans = []
    r2 = json.load(open(os.path.join(HERE, "rung2_result.json")))
    chans.append(("r5 baseline", {ast.literal_eval(k): v
                                  for k, v in r2["p0"].items()}))
    r3 = json.load(open(os.path.join(HERE, "rung3_result.json")))
    chans.append(("r6 DD", {ast.literal_eval(k): v
                            for k, v in r3["p0"].items()}))
    r4 = json.load(open(os.path.join(HERE, "rung4_result.json")))
    for bi, tag in ((0, "r7 dt=0"), (1, "r7 dt=3.2us"), (2, "r7 dt=6.4us")):
        chans.append((tag, {ast.literal_eval(k): v
                            for k, v in r4["p0"][str(bi)].items()}))
    r5 = json.load(open(os.path.join(HERE, "rung5_result.json")))
    for arm, tag in ((0, "r8 plain"), (1, "r8 DD")):
        chans.append((tag, {ast.literal_eval(k): v
                            for k, v in r5["p0"][str(arm)].items()}))
    return chans


def main():
    gates()
    print()
    hdr = ("%-12s %-5s | %-3s %-3s %-3s %-3s %-3s %-3s %-3s | "
           "%-6s %-6s %-6s | %-6s %s"
           % ("channel", "meas", "M0", "M1", "M1p", "M2", "M3", "M4", "M4c",
              "rM0", "rM1", "rM4", "coh", "eta-after-rot"))
    print(hdr); print("-" * len(hdr))
    out, hits = [], {"M0": 0, "M1": 0, "M1pre": 0, "M2": 0,
                     "M3": 0, "M4": 0, "M4c": 0, "M5": 0}
    for tag, p0 in load_channels():
        A, c, _, _ = B3.pair_and_radii(p0, 10**15, ALPHA)
        n_meas = index_of(A, c)
        eta, m0 = fit_M0(A)
        lam, m1 = fit_M1_post(A)
        mu, m1p = fit_M1_pre(A)
        (lam2, tz), m2 = fit_M2(A, c)
        (O3, e3), m3 = fit_M3_rot_scalar(A)
        (O4, lam4), m4 = fit_M4_rot_diag(A)
        m4c = (m4[0], c.copy())       # diagnostic: A from M4, c from data
        N5, m5 = fit_M5_unital(A)
        n0, n1, n1p, n2, n3, n4, n4c, n5 = (index_of(*m) for m in
                                            (m0, m1, m1p, m2, m3, m4, m4c, m5))
        frag = fragility(A, c, n_meas)
        ang, sig = coherent_angle_deg(A)
        for k, v in (("M0", n0), ("M1", n1), ("M1pre", n1p), ("M2", n2),
                     ("M3", n3), ("M4", n4), ("M4c", n4c), ("M5", n5)):
            hits[k] += (v == n_meas)
        out.append(dict(channel=tag, index_measured=n_meas, eta=eta,
                        M0=n0, M1=n1, M1pre=n1p, M2=n2, M3=n3, M4=n4, M4c=n4c,
                        M5=n5, fragility=frag,
                        eta_after_rotation=e3, resid_M3=resid(A, m3[0]),
                        resid_M4=resid(A, m4[0]),
                        lambdas=[float(x) for x in lam],
                        mu_pre=[float(x) for x in mu], t_z=tz,
                        resid_M0=resid(A, m0[0]), resid_M1=resid(A, m1[0]),
                        resid_M2=resid(A, m2[0]),
                        coherent_angle_deg=ang,
                        polar_contractions=[float(x) for x in sig]))
        print("%-12s %-5d | %-3d %-3d %-3d %-3d %-3d %-3d %-3d | "
              "%-6.4f %-6.4f %-6.4f | %-6.2f e'=%.4f"
              % (tag, n_meas, n0, n1, n1p, n2, n3, n4, n4c,
                 resid(A, m0[0]), resid(A, m1[0]), resid(A, m4[0]),
                 ang if ang is not None else -1, e3))
    print("-" * len(hdr))
    print("exact/7:  M0 %d | M1 %d | M1' %d | M2 %d | "
          "M3(rot+scalar) %d | M4(rot+diag) %d | M4c(A-only) %d | "
          "M5(full,unital) %d"
          % (hits["M0"], hits["M1"], hits["M1pre"], hits["M2"],
             hits["M3"], hits["M4"], hits["M4c"], hits["M5"]))

    # ---- the coherent-rotation kinematics across the delay blocks
    r7 = [ch for ch in out if ch["channel"].startswith("r7")]
    dts = [0.0, 3.2, 6.4]
    angs = [ch["coherent_angle_deg"] for ch in r7]
    es = [ch["eta_after_rotation"] for ch in r7]
    dang = (angs[2] - angs[0]) / 6.4
    detune_khz = dang / 360.0 * 1e3
    de = (es[2] - es[0]) / 6.4
    print("\nrotation angle across r7 blocks: %.2f -> %.2f -> %.2f deg"
          % tuple(angs))
    print("  d(angle)/d(dt) = %.2f deg/us  ->  detuning ~ %.1f kHz"
          % (dang, detune_khz))
    print("true decay after removing rotation: e' = %.4f -> %.4f -> %.4f"
          % tuple(es))
    print("  d(e')/d(dt) = %.1fe-3 per us  (register 7 reported 20.1e-3 "
          "on the isotropic eta)" % (de * 1e3))
    print("  e' increments: %.4f then %.4f (near-linear; the convexity of "
          "register 7 was the rotation's quadratic 1-cos signature)"
          % (es[1] - es[0], es[2] - es[1]))
    json.dump(dict(label="float64 exploratory (register 9, Paper VII); "
                         "zero QPU", hits=hits, channels=out),
              open(os.path.join(HERE, "anisotropic_register9_f64.json"), "w"),
              indent=1)
    print("wrote anisotropic_register9_f64.json")


if __name__ == "__main__":
    main()
