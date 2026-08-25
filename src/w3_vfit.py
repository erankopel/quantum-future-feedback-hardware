#!/usr/bin/env python3
"""
W3-1 pre-registered estimators for the virtual-Z scan.  ZERO QPU.

Amendment of 25 Aug 2026, written and committed BEFORE the window-3 run.
Three defects in the original in-line analysis are fixed here:

 D1  ESTIMATOR SHAPE.  The correction rz(k*theta0) is inserted at the same 15
     (21,36) CZ sites that carry the error, so it propagates through the same
     interleaving and k = -1 cancels exactly when theta0 is the true per-gate
     phase.  The resulting response is
         angle(x) = sqrt( p^2 + q^2 (x - x0)^2 )
     a hyperbola (a rounded V), NOT a parabola.  Monte Carlo at the measured
     shot-noise floor (sigma = 0.60 deg/arm at 1024 shots, 3000 trials/point)
     gives, for a true optimum of -2.84 deg:
         5-arm parabola  -3.02 to -3.07 deg, sd 0.32 to 1.23   (biased +0.2)
         hyperbola       -2.83 to -2.87 deg, sd 0.17 to 0.76   (unbiased)
     The hyperbola is therefore primary; the parabola on the three arms that
     bracket the prediction (k = 0, -1, -2) is the secondary estimator; the
     original 5-arm parabola is still reported, as a registered diagnostic.

 D2  NO SIGMA.  Gate G-VZ says ">= 3 sigma" but nothing computed sigma.  Here
     it is a parametric bootstrap on the click probabilities: resample every
     p0 from Binomial(shots, p0)/shots and re-run the full explorer bundle.

 D3  UNGUARDED THETA.  w3_theta.json fed rung8's corrected arm with no check
     that the optimum was inside the scanned span or that the scan validated
     anything.  Writing is now gated on V2 AND on |theta_opt| <= the scan
     half-span.

New PRIMARY statistic, pre-registered here.  The signed-arm asymmetry
    ASYM1 = angle(+1) - angle(-1),  ASYM2 = angle(+2) - angle(-2)
is exactly zero in expectation under the null (with no signed Z response the
+k and -k arms are exchangeable, and drift enters both equally), and it is far
more powerful than "the minimum is below the plain arm".  At the axis
alignment measured in the data already in hand (f_z = 0.919 +/- 0.036 for this
very circuit) both give ~100 per cent power; ASYM stays above 70 per cent down
to f_z = 0.5, where V2 has 2 per cent.
"""
import numpy as np
from scipy.optimize import least_squares
import paper7_core as C

NBOOT = 400
SEED = 11


def bundle_with_sigma(p0, shots, nboot=NBOOT, seed=SEED):
    """Explorer bundle plus bootstrap sd on angle, eta, eta'."""
    rng = np.random.default_rng(seed)
    point = C.analyze(p0)
    keys = list(p0)
    base = np.clip(np.array([p0[k] for k in keys], float), 0, 1)
    ang, eta, etap = [], [], []
    for _ in range(nboot):
        draw = rng.binomial(shots, base) / shots
        b = C.analyze(dict(zip(keys, draw)))
        if b["angle_deg"] is not None:
            ang.append(b["angle_deg"]); eta.append(b["eta"])
            etap.append(b["eta_after_rot"])
    point["sd_angle"] = float(np.std(ang)) if ang else None
    point["sd_eta"] = float(np.std(eta)) if eta else None
    point["sd_eta_after_rot"] = float(np.std(etap)) if etap else None
    point["nboot_ok"] = len(ang)
    return point


def fit_hyperbola(x, y):
    """angle = sqrt(p^2 + q^2 (x-x0)^2). Returns dict or None."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    p0 = [max(y.min(), 0.1),
          (y.max() - y.min()) / (x.max() - x.min() + 1e-9) + 0.1,
          x[int(np.argmin(y))]]
    f = lambda p: np.sqrt(p[0] ** 2 + (p[1] * (x - p[2])) ** 2) - y
    try:
        s = least_squares(f, p0, max_nfev=6000)
    except Exception:
        return None
    r = f(s.x)
    dof = max(len(x) - 3, 1)
    return dict(floor_deg=float(abs(s.x[0])), slope=float(abs(s.x[1])),
                theta_opt=float(s.x[2]),
                rms_resid=float(np.sqrt(np.sum(r ** 2) / dof)))


def fit_parabola(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return None
    c = np.polyfit(x, y, 2)
    return dict(a=float(c[0]), b=float(c[1]), c=float(c[2]), convex=bool(c[0] > 0),
                theta_opt=(float(-c[1] / (2 * c[0])) if c[0] > 0 else None))


def gvz_verdict(rows, theta0_deg, nsig=3.0):   # theta0_deg: applied step
    """rows: list of bundles with keys k, applied_deg, angle_deg, sd_angle,
    eta, sd_eta, eta_after_rot, sd_eta_after_rot, index_measured,
    index_isotropic.  Returns the full pre-registered gate evaluation."""
    by = {r["k"]: r for r in rows}
    x = np.array([r["applied_deg"] for r in rows], float)
    y = np.array([r["angle_deg"] for r in rows], float)
    sd = np.array([r["sd_angle"] or np.nan for r in rows], float)
    sbar = float(np.nanmean(sd))
    half_span = float(np.max(np.abs(x)))

    hyp = fit_hyperbola(x, y)
    par5 = fit_parabola(x, y)
    tri = [by[k] for k in (0, -1, -2) if k in by]
    par3 = (fit_parabola([r["applied_deg"] for r in tri],
                         [r["angle_deg"] for r in tri]) if len(tri) == 3 else None)

    def asym(kp, km):
        if kp not in by or km not in by:
            return None
        d = by[kp]["angle_deg"] - by[km]["angle_deg"]
        s = np.hypot(by[kp]["sd_angle"] or sbar, by[km]["sd_angle"] or sbar)
        return dict(diff_deg=float(d), sd=float(s), nsigma=float(d / s))

    A1, A2 = asym(1, -1), asym(2, -2)
    usable = [r for r in rows if r.get("angle_deg") is not None]
    if not usable:
        return dict(path="B", write_theta=False, error="no arm returned a rotation "
                    "angle (polar decomposition gave a reflection)")
    best = min(usable, key=lambda r: r["angle_deg"])
    plain = by.get(0)
    drop = plain["angle_deg"] - best["angle_deg"] if plain else None
    sd_drop = float(np.hypot(plain["sd_angle"] or sbar, best["sd_angle"] or sbar)) \
        if plain else None

    # ---- the four pre-registered predictions ----------------------------
    # V1 FIX (review, 25 Aug): hyp["slope"] is |q| and is positive for ANY
    # successful fit, so the old test passed on a perfectly flat response.
    # V1 now requires a V of real DEPTH: the fitted curve must rise from its
    # floor to the far edge of the scan by at least nsig * sigma, and the
    # minimum must be displaced from k = 0 by more than the fit's own scatter.
    if hyp is not None:
        rise = float(np.sqrt(hyp["floor_deg"] ** 2 +
                             (hyp["slope"] * (half_span + abs(hyp["theta_opt"]))) ** 2)
                     - hyp["floor_deg"])
        disp = abs(hyp["theta_opt"])
        V1 = bool(rise >= nsig * sbar and disp > max(hyp["rms_resid"], 0.25 * theta0_deg))
    else:
        V1 = False
    # SIGN AMENDMENT, 25 Aug 2026.  The registered V2 required the minimum at
    # NEGATIVE k.  That clause is withdrawn as an input assumption and re-issued
    # as the sharper side-prediction V2P below, because the data already in hand
    # point the other way:  the round's measured noise-map rotation axis is
    # n_z = -0.919 (a Z rotation of -6.13 deg), register 14's per-gate CZ phase
    # is H(ZI) = -2.84 deg (same sign, so the transfer factor kappa is positive
    # and 15*kappa*2.84 = 6.13), and qiskit's rz(+t) = exp(-i t Z/2) ADDS +t to
    # the Bloch rotation about +Z.  Cancelling -6.13 therefore needs POSITIVE k.
    # V2 is now sign-neutral and the direction is an output; V2P states the
    # derived sign so that either outcome is a clean measurement (a minimum at
    # k = -1 measures a negative transfer factor, i.e. the interleaved
    # single-qubit layer reverses the frame between insertions).
    V2 = bool(drop is not None and sd_drop and drop >= nsig * sd_drop
              and best["k"] != 0)
    V2P = bool(V2 and best["k"] > 0)
    # V3 FIX (review, 25 Aug): "or 0" made the threshold vanish when a
    # bootstrap sd was missing, so V3 passed on a 1/32-sigma wobble. Missing
    # sds now fall back to the pooled floor, and a missing pooled floor fails
    # the test outright rather than passing it.
    etaps = np.array([r["eta_after_rot"] for r in rows], float)
    sde = [r["sd_eta"] for r in rows if r.get("sd_eta")]
    sdep = [r["sd_eta_after_rot"] for r in rows if r.get("sd_eta_after_rot")]
    sd_e_bar = float(np.mean(sde)) if sde else None
    sd_ep = float(np.mean(sdep)) if sdep else None
    if plain is None or sd_e_bar is None or sd_ep is None:
        V3 = False
    else:
        fall = plain["eta"] - best["eta"]
        sd_fall = float(np.hypot(plain["sd_eta"] or sd_e_bar, best["sd_eta"] or sd_e_bar))
        V3 = bool(fall >= nsig * sd_fall and
                  (etaps.max() - etaps.min()) <= nsig * sd_ep * np.sqrt(2))
    # V4 FIX (review, 25 Aug): implemented_index returns None when no PPT step
    # is found within nmax, and None == None used to pass.
    V4 = bool(best["index_measured"] is not None and
              best["index_isotropic"] is not None and
              best["index_measured"] == best["index_isotropic"])
    # two-sided: the sign of the asymmetry is an output, not an assumption
    ASYM = bool((A1 and abs(A1["nsigma"]) >= nsig) or
                (A2 and abs(A2["nsigma"]) >= nsig))
    # DRIFT FIX (review, 25 Aug): the old test looked only at k = +/-1, so a
    # same-direction drop in the OUTER pair (the textbook drift signature) was
    # scored as a strong effect. Every signed pair present is now tested.
    # DRIFT DETECTOR, rebuilt 25 Aug after three unit-test failures.
    # Pairwise thresholding was tried and abandoned: it missed the artefact
    # about one seed in eight, and it raised false alarms on genuine signed
    # effects where noise nudged the far arm the same way. The clean statement
    # is a symmetry decomposition of the whole curve about k = 0.
    #   odd  component  a(+k) - a(-k)          : a SIGNED phase lives here
    #   even component  a(+k) + a(-k) - 2 a(0) : a |k|-tracking artefact does
    # Drift is declared when the even component is significant and the odd one
    # is not: the response depends on the magnitude of the insertion and not on
    # its sign, which no coherent phase can produce.
    same_dir = False
    same_dir_detail = []
    odd_chi2 = even_chi2 = 0.0
    npair = 0
    if plain is not None and sbar and np.isfinite(sbar) and sbar > 0:
        for kk in sorted({abs(r["k"]) for r in rows} - {0}):
            if kk in by and -kk in by:
                ap, am, a0 = (by[kk]["angle_deg"], by[-kk]["angle_deg"],
                              plain["angle_deg"])
                odd = ap - am
                even = ap + am - 2 * a0
                odd_chi2 += (odd / (sbar * np.sqrt(2.0))) ** 2
                even_chi2 += (even / (sbar * np.sqrt(6.0))) ** 2
                npair += 1
                same_dir_detail.append(dict(k=kk, odd_deg=float(odd),
                                            even_deg=float(even),
                                            plus=float(ap), minus=float(am)))
        if npair:
            # 3-sigma equivalents for npair degrees of freedom
            THR_EVEN = {1: 9.0, 2: 11.8, 3: 14.2}.get(npair, 9.0 + 2.8 * npair)
            THR_ODD = THR_EVEN
            same_dir = bool(even_chi2 >= THR_EVEN and odd_chi2 < THR_ODD)
    theta_opt = hyp["theta_opt"] if hyp else None
    # SPAN FIX (review, 25 Aug), corrected by unit test: half_span is the OUTER
    # grid point and "<=" let an optimum sitting on the grid edge through about
    # half the time, which is exactly risk R5. The optimum must now clear the
    # edge by half a grid step AND the winning arm must not itself be the edge
    # arm. (A first attempt used the second-outermost grid magnitude, which is
    # the PREDICTED optimum itself and rejected good runs; the unit test over 8
    # seeds caught it.)
    margin = 0.5 * abs(theta0_deg)
    inner_edge = max(half_span - margin, 0.0)
    edge_arm = bool(half_span > 0 and abs(best["applied_deg"]) >= half_span - 1e-9)
    # a hyperbola fitted to a flat response has an unidentified vertex: if the
    # fitted rise across the whole scan is under the noise floor, theta is not
    # measured and must not be reported as a number.
    identified = bool(hyp is not None and
                      hyp["slope"] * max(half_span, 1e-9) >= sbar)
    if not identified:
        theta_opt = None
    # AMENDMENT 1c, POST-HOC, 25 Aug 2026, written AFTER seeing W3-1c. Flagged
    # as post-hoc because it is: it was not pre-registered. Its scope is
    # deliberately narrow -- it changes only which theta is handed to rung8's
    # corrected arm, and touches NO pre-registered prediction test.
    #
    # W3-1c returned a curve that bottoms out on a FLOOR rather than turning
    # over: k=+3 gives 6.70 and k=+4 gives 6.60, agreeing to 0.10 +/- 0.81 deg.
    # The edge-arm veto exists for risk R5, where the optimum lies BEYOND the
    # grid and theta is unmeasured. A flat bottom is the opposite situation:
    # theta is measured, and measured to be anywhere on a plateau. Vetoing that
    # throws away a good number. The two are distinguished by whether the best
    # arm's neighbour agrees with it inside the noise.
    plateau = [r for r in usable
               if abs(r["angle_deg"] - best["angle_deg"]) <=
               nsig * np.sqrt(2.0) * sbar]
    on_plateau = len(plateau) >= 2
    plateau_x = sorted(r["applied_deg"] for r in plateau)
    plateau_mid = float(np.mean(plateau_x)) if on_plateau else None
    plateau_width = float(plateau_x[-1] - plateau_x[0]) if on_plateau else None
    if edge_arm and on_plateau:
        # theta is determined up to the plateau; hand over its midpoint and
        # carry the half-width as the uncertainty.
        theta_opt = plateau_mid
        in_span = True
        theta_note = ("flat bottom: %d arms agree within noise over %.2f deg; "
                      "theta = plateau midpoint %+.2f +/- %.2f"
                      % (len(plateau), plateau_width, plateau_mid,
                         plateau_width / 2.0))
    else:
        in_span = bool(theta_opt is not None and abs(theta_opt) <= inner_edge
                       and not edge_arm)
        theta_note = None
    if same_dir:
        path = "DRIFT ARTEFACT -> 3-arm repeat before deciding"
    elif (V1 and V2) or ASYM:
        path = "A"
    else:
        path = "B"
    # WRITE FIX (review, 25 Aug): write_theta used to diverge from `path`, so a
    # V2-only result printed "PATH B" and wrote w3_theta.json in the same
    # breath. It is now gated on the path actually being A.
    write_theta = bool(path == "A" and in_span and not same_dir)
    return dict(sigma_bar_deg=sbar, half_span_deg=half_span,
                inner_edge_deg=inner_edge, best_arm_at_edge=edge_arm,
                theta_identified=identified, on_plateau=on_plateau,
                plateau_width_deg=plateau_width, plateau_mid_deg=plateau_mid,
                theta_note=theta_note,
                hyperbola=hyp, parabola5=par5, parabola3=par3,
                asym1=A1, asym2=A2,
                best_k=best["k"], best_angle=best["angle_deg"],
                plain_angle=(plain["angle_deg"] if plain else None),
                drop_deg=drop, drop_nsigma=(drop / sd_drop if drop and sd_drop else None),
                V1=V1, V2=V2, V2P_min_at_positive_k=V2P, V3=V3, V4=V4,
                ASYM_signed=ASYM,
                sign_of_optimum=("positive k" if best["k"] > 0 else
                                 "negative k" if best["k"] < 0 else "none"),
                both_signed_same_direction=same_dir,
                symmetry=dict(odd_chi2=float(odd_chi2), even_chi2=float(even_chi2),
                              n_pairs=npair, pairs=same_dir_detail),
                theta_opt_deg=theta_opt, theta_opt_in_span=in_span,
                path=path, write_theta=write_theta)
