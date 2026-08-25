"""ZERO QPU. Pre-run power / estimator study for W3-1.

PHYSICAL RESPONSE MODEL. The correction rz(k*theta0) is inserted at the SAME
15 (21,36) CZ sites that carry the error, so it propagates through the same
interleaving: whatever the transfer factor kappa, k = -1 cancels the
gate-associated Z part exactly when theta0 equals the true per-gate phase.
Write the round's noise-map rotation as a Z-type part Theta_z and a residual
Theta_perp; then

    angle(k) = sqrt( Theta_perp^2 + (Theta_z * (1 + k * r))^2 ),
    r = theta0_true / theta0_applied   (r = 1 if register 14's number is right)

which is a HYPERBOLA (a rounded V), not a parabola. The pre-registered code
fits a PARABOLA to all five arms. This script measures what that costs.
"""
import numpy as np
from scipy.optimize import least_squares

SIG   = 0.60          # deg, measured shot-noise floor on the angle @1024 shots
TOT   = 6.67          # deg, measured baseline angle of the round (rung2 build)
TH0   = 2.84          # deg per gate, applied step
KS    = np.array([0,-1,1,-2,2], float)
X     = KS*TH0
rng   = np.random.default_rng(7)

def truth(k, fz, r=1.0):
    tz = TOT*fz; tp = np.sqrt(max(TOT**2-tz**2,0.0))
    return np.sqrt(tp**2 + (tz*(1+k*r))**2)

def fit_quad(x,y):
    c = np.polyfit(x,y,2)
    return (-c[1]/(2*c[0]) if c[0]>0 else np.nan), c[0]>0

def fit_hyp(x,y):
    p0=[max(y.min(),0.1), (y.max()-y.min())/(x.max()-x.min()+1e-9)+.1, x[np.argmin(y)]]
    f=lambda p: np.sqrt(p[0]**2+(p[1]*(x-p[2]))**2)-y
    try:
        s=least_squares(f,p0,max_nfev=4000)
        return s.x[2], abs(s.x[1])
    except Exception:
        return np.nan, np.nan

print("W3-1 POWER STUDY  (sigma=%.2f deg/arm @1024 shots, baseline angle %.2f deg)"%(SIG,TOT))
print("f_z = fraction of the round's rotation that is Z-type (gate-associated)\n")
hdr=("f_z","true min","V2 pass %","quad5 theta_opt","hyp theta_opt","quad3 theta_opt")
print("%5s %9s %10s %22s %22s %22s"%hdr)
NB=3000
for fz in (0.5,0.6,0.7,0.8,0.9,0.98):
    ytrue = truth(KS,fz)
    v2=0; q5=[]; hy=[]; q3=[]
    for _ in range(NB):
        y = ytrue + rng.normal(0,SIG,5)
        # V2: fitted minimum below the k=0 arm by >=3 sigma of the difference
        drop = y[0]-y.min(); 
        if drop >= 3*SIG*np.sqrt(2): v2+=1
        t,conv = fit_quad(X,y)
        if conv and np.isfinite(t): q5.append(t)
        th,_ = fit_hyp(X,y); 
        if np.isfinite(th): hy.append(th)
        m=[0,1,3]      # k = 0,-1,-2 : the three arms bracketing the prediction
        t3,c3 = fit_quad(X[m],y[m])
        if c3 and np.isfinite(t3): q3.append(t3)
    f=lambda v: ("%7.2f +/-%5.2f (n=%2d%%)"%(np.median(v),np.std(v),100*len(v)//NB)) if len(v)>NB//50 else "   unusable      "
    print("%5.2f %9.2f %9.0f%% %22s %22s %22s"%(fz,-TH0,100*v2/NB,f(q5),f(hy),f(q3)))

print("\nTRUE per-gate optimum is -2.84 deg in every row. Read the three")
print("estimator columns for bias: the 5-arm parabola is the pre-registered one.")
print("\nARM-BY-ARM predicted angles (deg), f_z = 0.8:")
for k in KS.astype(int):
    print("   k=%+d (applied %+6.2f deg/gate): %6.2f" % (k,k*TH0,truth(k,0.8)))
