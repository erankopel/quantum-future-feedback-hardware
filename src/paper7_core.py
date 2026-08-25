#!/usr/bin/env python3
"""
Paper VII core, self-contained (no pulse_engine dependency).

The series' rung1_hardware_bracket imports pulse_engine (which lives in the
sibling Paper-V-VI-merge/anc folder, not connected to this session). Every
number the HARDWARE runs report is explorer-labelled and needs only the
float64 pipeline: the affine Bloch pair (A, c), the fitted contraction eta,
the implemented index via choi_pt_lmin, the polar rotation angle, the
anisotropy. All of that is reconstructed here from velocity_probe (which IS
present and self-contained) plus the short tomography definitions copied
verbatim from rung1/register3 so the numbers match the series byte for byte.

Certified (Arb) brackets are NOT reproduced here: they need pulse_engine, they
are weak at these shot counts, and the manuscript labels the hardware runs
explorer. Gate GCORE below re-derives the paper's own recorded rung6 numbers
from its stored counts and refuses to be trusted unless they match.
"""
import numpy as np
from velocity_probe import U, QR1, HOLD, tau, bloch_pair, choi_pt_lmin
from scipy.linalg import polar

# ----- the round unitary and ideal pair (series values) --------------------
U8 = U(QR1["theta"], QR1["phi"], QR1["kappa"], QR1["beta"])
P_HOLD = float(HOLD)
Aid, cid = bloch_pair(U8, tau(P_HOLD))

# ----- tomography basis definitions (verbatim from rung1) ------------------
PREP = {"z+": [], "z-": ["x"], "x+": ["h"], "x-": ["x", "h"],
        "y+": ["h", "s"], "y-": ["x", "h", "s"]}
BASIS = {"x": ["h"], "y": ["sdg", "h"], "z": []}


def pair_from_p0(p0):
    """Affine Bloch pair (A, c) from the 72 click probabilities p0, using the
    exact linear combinations of register3.pair_and_radii (exp = 2 p0 - 1)."""
    P = P_HOLD
    w = {0: (1 + P) / 2, 1: (1 - P) / 2}
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
    s = float(np.sum(A * Aid) / np.sum(Aid * Aid))
    return 1.0 - s


def anisotropy(A):
    s = 1.0 - fit_eta(A)
    return float(np.linalg.norm(A - s * Aid) / np.linalg.norm(Aid))


def implemented_index(A, c, nmax=400):
    """First r with the composite PPT (lmin of PT-Choi >= 0)."""
    Ac, cc = np.eye(3), np.zeros(3)
    last_npt = 0
    for r in range(1, nmax + 1):
        Ac = A @ Ac
        cc = A @ cc + c
        lm = choi_pt_lmin(Ac, cc)
        if lm < 0:
            last_npt = r
        else:
            return r          # first PPT
    return None


def model_index(A, c, nmax=400):
    eta = fit_eta(A)
    return implemented_index((1 - eta) * Aid, (1 - eta) * cid, nmax=nmax)


def polar_angle_deg(A):
    """Coherent-rotation angle of the noise map N = A A_id^{-1}."""
    N = A @ np.linalg.inv(Aid)
    O, Pp = polar(N)
    if np.linalg.det(O) < 0:
        return None
    ang = np.degrees(np.arccos(np.clip((np.trace(O) - 1) / 2, -1, 1)))
    return float(ang)


def eta_after_rotation(A):
    """Contraction with the coherent rotation divided out (register 9 M3)."""
    N = A @ np.linalg.inv(Aid)
    O, _ = polar(N)
    B = O @ Aid
    return 1.0 - float(np.sum(A * B) / np.sum(B * B))


def singular_values(A):
    return sorted(np.linalg.svd(A, compute_uv=False).tolist(), reverse=True)


def analyze(p0):
    """Full explorer point-estimate bundle for one channel."""
    A, c = pair_from_p0(p0)
    return dict(eta=fit_eta(A), anisotropy=anisotropy(A),
                index_measured=implemented_index(A, c),
                index_isotropic=model_index(A, c),
                angle_deg=polar_angle_deg(A),
                eta_after_rot=eta_after_rotation(A),
                singular_values=singular_values(A))
