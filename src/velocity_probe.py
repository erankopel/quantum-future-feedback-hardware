#!/usr/bin/env python3
"""
Paper VI register 1: log-margin velocities along the operating schedules.

Explorer computations in float64 (labelled as such; certification comes later
per the seed's discipline). The model is a numpy mirror of the valve half of the merged Paper V (former Paper VI)
pulse_engine.py (anc/pulse_engine.py in the merged V+VI package), gated
against its published atlas values before any new number is trusted.

Definitions (seed, RATE FORM): along a fixed schedule, track
lmin(r) = smallest eigenvalue of the partial-transposed Choi state of the
first r rounds' composite. The log-margin velocity of a round is
D_r = log|lmin(r)| - log|lmin(r-1)|. Per-round velocities are then averaged
over round TYPE on the actual operating schedules (seed guardrail:
non-commutativity makes velocities schedule-dependent, so they are defined
on the schedules, not on homogeneous intuition).

  v_in  : mean D over window (in-window) rounds  -> drift into PPT interior
  v_out : mean D over hot (out-of-window) rounds -> drift toward NPT

Outputs paper6/velocity_register1_f64.json and a console table.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------ model (float64)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
SIG = [X, Y, Z]
I2 = np.eye(2, dtype=complex)
I8 = np.eye(8, dtype=complex)


def ry(a):
    return np.array([[np.cos(a / 2), -np.sin(a / 2)],
                     [np.sin(a / 2), np.cos(a / 2)]], complex)


# SWAP of factors M and L, F fixed (mirrors pulse_engine.py index order (M,F,L))
SW = np.zeros((8, 8), complex)
for m in range(2):
    for f in range(2):
        for l in range(2):
            SW[(l * 2 + f) * 2 + m, (m * 2 + f) * 2 + l] = 1
ZY = np.kron(I2, np.kron(Z, Y))


def U(th, ph, k, b):
    P0 = np.diag([1, 0]).astype(complex)
    P1 = np.diag([0, 1]).astype(complex)
    UW = np.kron(P0, np.kron(ry(np.pi - 2 * th), I2)) + \
         np.kron(P1, np.kron(ry(2 * th), I2))
    Uw = np.cos(k / 2) * I8 - 1j * np.sin(k / 2) * ZY
    Uf = np.cos(ph) * I8 - 1j * np.sin(ph) * SW
    return np.kron(ry(b), np.kron(I2, I2)) @ Uf @ Uw @ UW


def tau(p):
    return 0.5 * I2 + 0.5 * p * Z


def ptr_FL(R):
    out = np.zeros((2, 2), complex)
    for i in range(2):
        for j in range(2):
            out[i, j] = sum(R[i * 4 + a, j * 4 + a] for a in range(4))
    return out


def bloch_pair(u, anc1):
    anc = np.kron(anc1, anc1)

    def Phi(rho):
        return ptr_FL(u @ np.kron(rho, anc) @ u.conj().T)

    c = np.array([np.trace(SIG[i] @ Phi(0.5 * I2)).real for i in range(3)])
    A = np.zeros((3, 3))
    for j in range(3):
        Ps = Phi(SIG[j])
        for i in range(3):
            A[i, j] = (0.5 * np.trace(SIG[i] @ Ps)).real
    return A, c


def choi_pt_lmin(A, c):
    """lambda_min of the partial-transposed Choi of the affine pair (A, c)."""
    J = np.zeros((4, 4), complex)
    for i in range(2):
        for j in range(2):
            E = np.zeros((2, 2), complex); E[i, j] = 1
            t = np.trace(E) / 2
            v = np.array([np.trace(SIG[k] @ E) / 2 for k in range(3)])
            w = t * c + A @ v
            M = t * I2 + sum(w[k] * SIG[k] for k in range(3))
            J += np.kron(E, M) / 2
    H = J.reshape(2, 2, 2, 2).transpose(2, 1, 0, 3).reshape(4, 4)
    return float(np.linalg.eigvalsh(H)[0].real)


# ------------------------------------------------------------ targets
QR1 = dict(theta=0.8762894250210506, phi=0.044736280102247346,
           kappa=0.1247057659254267, beta=0.05210240689909829)
QC1 = dict(theta=0.6891907806455668, phi=0.044736280102247346,
           kappa=0.11191302043215262, beta=0.05538719740686704)
HOLD = 0.4016      # water-stretch operating polarisation
DEPTH = 0.05       # hot pulse depth used in the certified minimal pulse


def trajectory(u, sched):
    """lmin(r) for r = 1..len(sched), composing earliest-first."""
    A = np.eye(3)
    c = np.zeros(3)
    out = []
    for Ai, ci in sched:
        A = Ai @ A
        c = Ai @ c + ci
        out.append(choi_pt_lmin(A, c))
    return out


def centred_sched(u, n, k, hold, depth):
    ph = bloch_pair(u, tau(hold))
    pd = bloch_pair(u, tau(depth))
    a = (n - k) // 2
    return [ph] * a + [pd] * k + [ph] * (n - k - a), a


def velocities(traj, a, k):
    """Split mean per-round log-margin velocities by round type.
    a hold rounds, then k pulse rounds, then the rest hold."""
    d = [traj[i] for i in range(len(traj))]
    logm = [np.log(abs(x)) for x in d]
    D = [logm[i] - logm[i - 1] for i in range(1, len(logm))]
    # D[i] is the velocity of round i+1 (1-based round index i+1)
    early = D[0:a - 1]            # rounds 2..a
    pulse = D[a - 1:a + k - 1]    # rounds a..a+k
    late = D[a + k - 1:]          # rounds a+k+1..n
    return dict(v_pulse=float(np.mean(pulse)),
                v_hold_early=float(np.mean(early)),
                v_hold_late=float(np.mean(late)),
                D=[float(x) for x in D],
                lmin=[float(x) for x in traj])


def homogeneous_velocity(u, n, p):
    pair = bloch_pair(u, tau(p))
    traj = trajectory(u, [pair] * n)
    logm = [np.log(abs(x)) for x in traj]
    D = [logm[i] - logm[i - 1] for i in range(1, len(logm))]
    return float(np.mean(D)), traj


def k_min_scan(u, n, hold, depth, mode="open"):
    """First centred pulse length k that switches the composite.
    mode 'open': hold in-window, switch = lmin(n) < 0 (NPT).
    mode 'close': hold hot, switch = lmin(n) > 0 (PPT)."""
    want_neg = (mode == "open")
    for k in range(1, n):
        sched, _ = centred_sched(u, n, k, hold, depth)
        lm = trajectory(u, sched)[-1]
        if (lm < 0) == want_neg:
            return k
    return None


# ------------------------------------------------------------ gates first
def main():
    report = {"label": "float64 exploratory (register 1, Paper VI)",
              "gates": {}, "qr1": {}, "qc1": {}}
    fails = 0

    U_QR1 = U(QR1["theta"], QR1["phi"], QR1["kappa"], QR1["beta"])
    U_QC1 = U(QC1["theta"], QC1["phi"], QC1["kappa"], QC1["beta"])

    # G1: unpulsed QR1 hold must reproduce the atlas lmin +1.3011396e-3
    pair_hold_qr1 = bloch_pair(U_QR1, tau(HOLD))
    lm = trajectory(U_QR1, [pair_hold_qr1] * 88)[-1]
    ok = abs(lm - 0.0013011396385356213) < 1e-9
    report["gates"]["G1_qr1_unpulsed_lmin"] = {"value": lm, "atlas": 0.0013011396385356213, "ok": ok}
    fails += not ok

    # G2: certified minimal pulse: centred k=8 switches, k=7 does not
    lm8 = trajectory(U_QR1, centred_sched(U_QR1, 88, 8, HOLD, DEPTH)[0])[-1]
    lm7 = trajectory(U_QR1, centred_sched(U_QR1, 88, 7, HOLD, DEPTH)[0])[-1]
    ok = (lm8 < 0) and (lm7 > 0)
    report["gates"]["G2_minimal_pulse"] = {"lmin_k8": lm8, "lmin_k7": lm7, "ok": ok}
    fails += not ok

    # G3: QC1 unpulsed hold (n=80) is PPT, ~+7.7e-4 per register 2
    pair_hold_qc1 = bloch_pair(U_QC1, tau(HOLD))
    lm = trajectory(U_QC1, [pair_hold_qc1] * 80)[-1]
    ok = (lm > 0) and abs(lm - 7.7e-4) < 5e-5
    report["gates"]["G3_qc1_unpulsed"] = {"value": lm, "ok": ok}
    fails += not ok

    if fails:
        print("GATES FAILED - refusing to report ungated numbers.")
        print(json.dumps(report["gates"], indent=1))
        raise SystemExit(1)
    print("gates: G1 G2 G3 all OK (float64 pipeline mirrors the certified engine)\n")

    # ------------------------------------------------ QR1 mode OPEN (hold in window, hot pulse)
    for (k, label) in ((12, "k=12 (signal point k_min+4)"),):
        sched, a = centred_sched(U_QR1, 88, k, HOLD, DEPTH)
        v = velocities(trajectory(U_QR1, sched), a, k)
        report["qr1"]["open"] = dict(schedule=f"hold {HOLD}, centred {k} at {DEPTH}, n=88",
                                     **{kk: vv for kk, vv in v.items() if kk != "D" and kk != "lmin"},
                                     profile=v["D"], lmin_traj=v["lmin"])

    # homogeneous velocities at the two operating bath values (QR1)
    vh, _ = homogeneous_velocity(U_QR1, 88, HOLD)
    vp, _ = homogeneous_velocity(U_QR1, 88, DEPTH)
    report["qr1"]["homogeneous"] = {"v_hold_0.4016": vh, "v_pulse_0.05": vp}

    # ------------------------------------------------ QR1 mode CLOSE (hot hold, window pulse)
    for (k, wdepth) in ((25, 0.556), (35, HOLD)):
        sched, a = centred_sched(U_QR1, 88, k, DEPTH, wdepth)
        v = velocities(trajectory(U_QR1, sched), a, k)
        report["qr1"][f"close_{wdepth}"] = dict(
            schedule=f"hold {DEPTH}, centred {k} at {wdepth}, n=88",
            **{kk: vv for kk, vv in v.items() if kk != "D" and kk != "lmin"})

    # ------------------------------------------------ QC1 mode OPEN
    kqc1 = 9  # k_min+4 at depth 0.05 (k_min = 5 per register 2)
    sched, a = centred_sched(U_QC1, 80, kqc1, HOLD, DEPTH)
    v = velocities(trajectory(U_QC1, sched), a, kqc1)
    report["qc1"]["open"] = dict(schedule=f"hold {HOLD}, centred {kqc1} at {DEPTH}, n=80",
                                 **{kk: vv for kk, vv in v.items() if kk != "D" and kk != "lmin"},
                                 profile=v["D"], lmin_traj=v["lmin"])
    vh, _ = homogeneous_velocity(U_QC1, 80, HOLD)
    vp, _ = homogeneous_velocity(U_QC1, 80, DEPTH)
    report["qc1"]["homogeneous"] = {"v_hold_0.4016": vh, "v_pulse_0.05": vp}

    # QC1 CLOSE (mirror: hot hold, pulse into window at the water value)
    sched, a = centred_sched(U_QC1, 80, 25, DEPTH, HOLD)
    v = velocities(trajectory(U_QC1, sched), a, 25)
    report["qc1"]["close_0.4016"] = dict(
        schedule=f"hold {DEPTH}, centred 25 at {HOLD}, n=80",
        **{kk: vv for kk, vv in v.items() if kk != "D" and kk != "lmin"})

    # ------------------------------------------------ k_min scans (atlas verification + QC1 CLOSE)
    report["k_min_checks"] = {
        "qr1_open_0.05": k_min_scan(U_QR1, 88, HOLD, DEPTH, "open"),
        "qc1_open_0.05": k_min_scan(U_QC1, 80, HOLD, DEPTH, "open"),
        "qr1_close_0.556": k_min_scan(U_QR1, 88, DEPTH, 0.556, "close"),
        "qc1_close_0.4016": k_min_scan(U_QC1, 80, DEPTH, HOLD, "close"),
    }

    out = os.path.join(HERE, "velocity_register1_f64.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)

    # ------------------------------------------------ console summary
    def row(tag, d):
        print("%-34s v_pulse=%+.4f  v_hold(early)=%+.4f  v_hold(late)=%+.4f"
              % (tag, d["v_pulse"], d["v_hold_early"], d["v_hold_late"]))

    print("PER-ROUND LOG-MARGIN VELOCITIES (float64, on-schedule)")
    row("QR1 OPEN  (k=12@0.05 in 0.4016)", report["qr1"]["open"])
    row("QR1 CLOSE (k=25@0.556 in 0.05)", report["qr1"]["close_0.556"])
    row("QR1 CLOSE (k=35@0.4016 in 0.05)", report["qr1"]["close_0.4016"])
    row("QC1 OPEN  (k=9@0.05 in 0.4016)", report["qc1"]["open"])
    row("QC1 CLOSE (k=25@0.4016 in 0.05)", report["qc1"]["close_0.4016"])
    print()
    print("homogeneous (pure-schedule) velocities:")
    print("  QR1: v(0.4016)=%+.4f/round   v(0.05)=%+.4f/round"
          % (report["qr1"]["homogeneous"]["v_hold_0.4016"],
             report["qr1"]["homogeneous"]["v_pulse_0.05"]))
    print("  QC1: v(0.4016)=%+.4f/round   v(0.05)=%+.4f/round"
          % (report["qc1"]["homogeneous"]["v_hold_0.4016"],
             report["qc1"]["homogeneous"]["v_pulse_0.05"]))
    print()
    print("k_min scan checks (vs atlas):", json.dumps(report["k_min_checks"]))

    # velocity profile structure along the circuit (QR1 OPEN k=12)
    prof = report["qr1"]["open"]["profile"]
    traj = report["qr1"]["open"]["lmin_traj"]
    n = len(prof)
    third = n // 3
    print("\nVELOCITY PROFILE STRUCTURE (QR1 OPEN, k=12 centred; pulse rounds 39-50)")
    print("  D_r by circuit third: early %+.4f | mid %+.4f | late %+.4f"
          % (np.mean(prof[:third]), np.mean(prof[third:2*third]), np.mean(prof[2*third:])))
    imin = int(np.argmin(np.abs(traj)))
    print("  lmin closest to zero at round %d (value %+.3e)" % (imin + 1, traj[imin]))
    print("  lmin(1)=%+.3e  lmin(88)=%+.4e" % (traj[0], traj[-1]))
    print("  sample D_r:", " ".join("%+.3f" % prof[i] for i in (0, 10, 20, 30, 37, 38, 39, 44, 49, 50, 51, 60, 70, 80, 86)))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
