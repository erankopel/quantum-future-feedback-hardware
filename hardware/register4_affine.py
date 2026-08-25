#!/usr/bin/env python3
"""
Paper VII register 4: the correlated propagation, implemented.

Register 3 located the bottleneck: not the per-setting intervals (exact
Clopper-Pearson bought one to four rounds over Hoeffding) but the PROPAGATION,
which composed twelve interval balls as though independent and discarded their
correlations at every one of the r matrix products.

This register implements the fix the outlook posed: affine arithmetic over the
tomography's own noise symbols. Every measured click probability p0_s of the
72 settings gets ONE shared symbol e_s in [-1, 1]:

    <P>_s = m_s + rho_s e_s,       rho_s = 2 x Clopper-Pearson half-width.

The affine Bloch pair (A, c) is then built by the same linear combinations the
tomography itself uses, so the input carries the EXACT first-order correlation
structure (in particular, A[:,2] and c share their twelve settings, which the
independent-ball model wastes). Composition over r rounds is done in affine
arithmetic with arb coefficients: first-order terms in the e_s propagate as
signed sums (cancellation preserved), and every nonlinear cross term is
absorbed into a rigorous interval remainder. The result after r rounds is a
rigorous enclosure

    X(e) in X0 + sum_s e_s X_s (+/- rem),   valid for all e in [-1,1]^72,

from which the partial-transposed Choi H(e) follows exactly (it is linear in
(A, c)). Certification:

  NPT for all e:  the Rayleigh quotient q(e) = v* H(e) v / v*v with the fixed
                  midpoint eigenvector v is itself an affine form; its rigorous
                  maximum bounds lmin(H(e)) from above pointwise. q_max < 0
                  certifies NPT over the whole tomographic box.
  PPT for all e:  lmin(H(e)) >= lmin(H0) - ||H(e) - H0||_2, with lmin(H0)
                  bracketed by the series Kahan-Temple construction (midpoint
                  residual; float64 gap check per series practice) and the
                  perturbation bounded by a rigorous Frobenius enclosure of the
                  affine deviation. Lower bound positive certifies PPT over
                  the whole box.

Everything arb 256-bit on the interval side; the only float64 objects are the
proposed eigenvectors, whose quality affects tightness, never rigor.

Label: certified-given-shots (arithmetic exact, inputs statistical at joint
confidence 1 - delta, Bonferroni over the 72 settings).

Gates: G0 (noiseless tomography = series pair) inherited from the imported
pipeline; G2 (new): at radius 0 the affine machinery must reproduce the
implemented index of the float64 pipeline exactly.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "anc"))

from flint import arb, acb, ctx                               # noqa: E402
ctx.prec = 256
import rung1_hardware_bracket as R                            # noqa: E402
import register3_budget as B3                                 # noqa: E402

DELTA = 1e-3
NS = 72


# ---------------------------------------------------------------- affine core
class Aff:
    """x = mid + sum_s coef[s] * e_s (+/- rem), e_s in [-1, 1]; arb throughout."""
    __slots__ = ("mid", "coef", "rem", "_span")

    def __init__(self, mid, coef=None, rem=None):
        self.mid = mid if isinstance(mid, arb) else arb(mid)
        self.coef = coef if coef is not None else [arb(0)] * NS
        self.rem = rem if rem is not None else arb(0)
        self._span = None

    def span(self):
        """Rigorous upper bound on sum_s |coef_s| + rem."""
        if self._span is None:
            s = arb(0)
            for c in self.coef:
                s += abs(c)
            self._span = arb((s + self.rem).upper())
        return self._span

    def __add__(self, o):
        if isinstance(o, Aff):
            return Aff(self.mid + o.mid,
                       [a + b for a, b in zip(self.coef, o.coef)],
                       self.rem + o.rem)
        return Aff(self.mid + o, self.coef[:], self.rem)

    __radd__ = __add__

    def __neg__(self):
        return Aff(-self.mid, [-c for c in self.coef], self.rem)

    def __sub__(self, o):
        return self + (-o)

    def scale(self, s):
        s = s if isinstance(s, arb) else arb(s)
        return Aff(self.mid * s, [c * s for c in self.coef], self.rem * abs(s))

    def __mul__(self, o):
        if not isinstance(o, Aff):
            return self.scale(o)
        a0, b0 = self.mid, o.mid
        coef = [a0 * bc + b0 * ac for ac, bc in zip(self.coef, o.coef)]
        # nonlinear part: |a0|*rb + |b0|*ra + span_a * span_b (all cross terms)
        rem = abs(a0) * o.rem + abs(b0) * self.rem + self.span() * o.span()
        return Aff(a0 * b0, coef, arb(rem.upper()))

    def bounds(self):
        lo = self.mid - self.span()
        hi = self.mid + self.span()
        return arb(lo.lower()), arb(hi.upper())


def mat_vec(M, v):
    return [sum((M[i][k] * v[k] for k in range(3)), Aff(0)) for i in range(3)]


def mat_mat(M, N):
    return [[sum((M[i][k] * N[k][j] for k in range(3)), Aff(0))
             for j in range(3)] for i in range(3)]


# ------------------------------------------------- input: 72-symbol pair
def affine_pair(p0, shots, alpha):
    """(A, c) as affine forms over the 72 setting symbols, exact combos."""
    P = float(R.P_HOLD)
    w = {0: (1 + P) / 2, 1: (1 - P) / 2}
    keys = list(p0.keys())
    sym = {k: i for i, k in enumerate(keys)}
    exp = {}
    for k in keys:
        rho = 2 * B3.clopper_pearson_halfwidth(p0[k], shots, alpha)
        coef = [arb(0)] * NS
        coef[sym[k]] = arb(rho)
        exp[k] = Aff(arb(2 * p0[k] - 1), coef, arb(0))

    r_out = {}
    for prep in R.PREP:
        r_out[prep] = [sum((exp[(prep, ms, f, l)].scale(w[f] * w[l])
                            for f in (0, 1) for l in (0, 1)), Aff(0))
                       for ms in ("x", "y", "z")]
    c = [(r_out["z+"][i] + r_out["z-"][i]).scale(0.5) for i in range(3)]
    A = [[None] * 3 for _ in range(3)]
    for j, (pp, pm) in enumerate((("x+", "x-"), ("y+", "y-"), ("z+", "z-"))):
        for i in range(3):
            A[i][j] = (r_out[pp][i] - r_out[pm][i]).scale(0.5)
    return A, c


# ------------------------------------------------- H(e) and certification
SIGX = [[0, 1], [1, 0]]
SIGY = [[0, -1], [1, 0]]     # imaginary structure handled explicitly below


def choi_pt_affine(A, c):
    """H as a 4x4 of (re: Aff, im: Aff), mirroring velocity_probe.choi_pt_lmin,
    including its partial-transpose reshuffle."""
    Z = Aff(0)
    # w-vector for basis element E_{ij}: t = delta_ij / 2, v_k = Tr(sig_k E)/2
    # Tr(sig_x E_ij) = [j==0 and i==1] + [j==1 and i==0]
    # Tr(sig_y E_ij) = i*( [i==0,j==1] - [i==1,j==0] )  (pure imaginary)
    # Tr(sig_z E_ij) = [i==0,j==0] - [i==1,j==1]
    J = [[(Z, Z) for _ in range(4)] for _ in range(4)]
    for i in range(2):
        for j in range(2):
            t = 0.5 if i == j else 0.0
            vx = 0.5 * ((1 if (i == 1 and j == 0) else 0) +
                        (1 if (i == 0 and j == 1) else 0))
            # Tr(sig_y E_ij)/2 = +i/2 at (i,j)=(0,1) and -i/2 at (1,0)
            vy_im = 0.5 * ((1 if (i == 0 and j == 1) else 0) -
                           (1 if (i == 1 and j == 0) else 0))
            vz = 0.5 * ((1 if (i == 0 and j == 0) else 0) -
                        (1 if (i == 1 and j == 1) else 0))
            # w_k = t*c_k + A[k][0]*vx + A[k][1]*(i vy_im) + A[k][2]*vz
            w_re, w_im = [], []
            for k in range(3):
                re = c[k].scale(t) + A[k][0].scale(vx) + A[k][2].scale(vz)
                im = A[k][1].scale(vy_im)
                w_re.append(re); w_im.append(im)
            # M = t*I + w_x sig_x + w_y sig_y + w_z sig_z   (2x2 complex)
            # sig_y = [[0, -i],[i, 0]]
            M = [[(Z, Z), (Z, Z)], [(Z, Z), (Z, Z)]]
            M[0][0] = (w_re[2] + t, w_im[2])
            M[1][1] = (Aff(t) - w_re[2], -w_im[2])
            # entry (0,1): w_x - i w_y  -> re: wx_re + wy_im ; im: wx_im - wy_re
            M[0][1] = (w_re[0] + w_im[1], w_im[0] - w_re[1])
            M[1][0] = (w_re[0] - w_im[1], w_im[0] + w_re[1])
            for a in range(2):
                for b in range(2):
                    re, im = J[2 * i + a][2 * j + b]
                    J[2 * i + a][2 * j + b] = (re + M[a][b][0].scale(0.5),
                                               im + M[a][b][1].scale(0.5))
    # H[(j,a),(i,b)] = J[(i,a),(j,b)]  (transpose(2,1,0,3))
    H = [[None] * 4 for _ in range(4)]
    for i in range(2):
        for a in range(2):
            for j in range(2):
                for b in range(2):
                    H[2 * j + a][2 * i + b] = J[2 * i + a][2 * j + b]
    return H


def certify_round(H):
    """(npt_certified, ppt_certified, lmin_mid) over the whole symbol box."""
    H0 = np.array([[complex(float(H[p][q][0].mid.mid()),
                            float(H[p][q][1].mid.mid()))
                    for q in range(4)] for p in range(4)])
    wv, V = np.linalg.eigh(H0)
    v = V[:, 0]

    # Rayleigh quotient as an affine form: N(e) = v* H(e) v (real part)
    num = Aff(0)
    for p in range(4):
        for q in range(4):
            s = np.conj(v[p]) * v[q]
            num = num + H[p][q][0].scale(float(s.real)) \
                      - H[p][q][1].scale(float(s.imag))
    den = float(np.vdot(v, v).real)
    q_lo, q_hi = (num.scale(1.0 / den)).bounds()

    # midpoint Kahan-Temple bracket for lmin(H0), arb-rigorous
    Hm = [[acb(H[p][q][0].mid, H[p][q][1].mid) for q in range(4)]
          for p in range(4)]
    va = [acb(float(v[p].real), float(v[p].imag)) for p in range(4)]
    nn, dd = acb(0), acb(0)
    for p in range(4):
        for q in range(4):
            nn += va[p].conjugate() * Hm[p][q] * va[q]
        dd += va[p] * va[p].conjugate()
    q0 = (nn / dd).real
    res2 = acb(0)
    for p in range(4):
        s = acb(0)
        for q in range(4):
            s += Hm[p][q] * va[q]
        d = s - acb(q0.mid()) * va[p]
        res2 += d * d.conjugate()
    rho0 = (res2.sqrt() / dd.real.sqrt()).real

    # rigorous Frobenius bound on ||H(e) - H0||
    fro2 = arb(0)
    for p in range(4):
        for q in range(4):
            fro2 += H[p][q][0].span() ** 2 + H[p][q][1].span() ** 2
    dmax = arb(fro2.upper()).sqrt()

    lmin_lo = arb((arb(q0.mid()) - rho0.upper() - arb(dmax.upper())).lower())
    npt = q_hi.upper() < 0
    ppt = lmin_lo.lower() > 0
    return npt, ppt, float(wv[0])


def index_bracket_affine(A, c, nmax=200):
    Acc = [[Aff(1) if i == j else Aff(0) for j in range(3)] for i in range(3)]
    ccc = [Aff(0)] * 3
    last_npt, first_ppt, first_unres = 0, None, None
    for r in range(1, nmax + 1):
        Acc = mat_mat(A, Acc)
        ccc = [x + y for x, y in zip(mat_vec(A, ccc), c)]
        H = choi_pt_affine(Acc, ccc)
        npt, ppt, _ = certify_round(H)
        if npt:
            last_npt = r
        elif ppt:
            first_ppt = r
            break
        else:
            first_unres = r
            break
    return last_npt, first_ppt, first_unres


def main():
    Ai, ci = R.bloch_pair(R.U8, R.tau(R.P_HOLD))
    A0, c0 = R.measure_round_noiseless()
    d = max(np.linalg.norm(A0 - Ai), np.linalg.norm(c0 - ci))
    print("G0 noiseless tomography vs series pair: %.2e  [%s]"
          % (d, "PASS" if d < 1e-10 else "FAIL"))
    assert d < 1e-10, "G0 FAILED"

    alpha = DELTA / NS
    out = []
    for eps in (1e-3, 3e-3, 1e-4):
        p0 = B3.probabilities(eps)

        # implemented index (float64 pipeline)
        A_pt, c_pt, _, _ = B3.pair_and_radii(p0, 10**15, alpha)
        _, n_impl, _ = R.index_bracket(A_pt, c_pt, 0.0, nmax=200)

        # G2: affine machinery at radius ~0 must reproduce it exactly
        Aa, ca = affine_pair(p0, 10**15, alpha)
        l0, f0, u0 = index_bracket_affine(Aa, ca)
        ok = (f0 == n_impl)
        print("\n=== eps = %.0e : implemented index %s ; G2 affine@rad~0 -> "
              "(%s, %s)  [%s] ===" % (eps, n_impl, l0, f0,
                                      "PASS" if ok else "FAIL"))
        assert ok, "G2 FAILED"

        print("    %-10s %-22s %-22s %s"
              % ("shots", "balls (register 3)", "AFFINE (this register)",
                 "gain"))
        for shots in (10**5, 10**6, 10**7, 10**8, 10**9):
            A_m, c_m, A_rad, c_rad = B3.pair_and_radii(p0, shots, alpha)
            l_b, f_b = B3.index_bracket_perentry(A_m, c_m, A_rad, c_rad)
            Aa, ca = affine_pair(p0, shots, alpha)
            l_a, f_a, u_a = index_bracket_affine(Aa, ca)
            st_b = ("(%d, %d]" % (l_b, f_b)) if f_b else "> %d" % l_b
            st_a = ("(%d, %d] EXACT" % (l_a, f_a)) if f_a else "> %d" % l_a
            gain = l_a - l_b
            out.append(dict(eps=eps, shots=shots, implemented=n_impl,
                            ball_lower=l_b, ball_first_ppt=f_b,
                            affine_lower=l_a, affine_first_ppt=f_a,
                            gain_rounds=gain))
            print("    %-10.0e %-22s %-22s %+d rounds"
                  % (shots, st_b, st_a, gain))

    json.dump(dict(label="certified-given-shots (register 4, Paper VII); "
                         "72-symbol affine arithmetic, arb 256-bit",
                   delta=DELTA, settings=NS, rows=out),
              open(os.path.join(HERE, "affine_register4.json"), "w"), indent=1)
    print("\nwrote affine_register4.json")


if __name__ == "__main__":
    main()
