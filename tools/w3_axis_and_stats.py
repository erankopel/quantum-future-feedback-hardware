"""ZERO QPU. (A) Estimate f_z -- how Z-type the round's coherent rotation
already is -- from data ALREADY IN HAND, because f_z sets W3-1's power.
(B) Compare candidate test statistics for the G-VZ gate."""
import json, ast, numpy as np
from scipy.linalg import polar
import paper7_core as C

rng = np.random.default_rng(3)

def axis_angle(A):
    N = A @ np.linalg.inv(C.Aid)
    O, _ = polar(N)
    if np.linalg.det(O) < 0: return None, None
    ang = np.degrees(np.arccos(np.clip((np.trace(O)-1)/2, -1, 1)))
    w = np.array([O[2,1]-O[1,2], O[0,2]-O[2,0], O[1,0]-O[0,1]])
    n = np.linalg.norm(w)
    return ang, (w/n if n > 1e-12 else None)

def load(f, arm=None):
    d = json.load(open(f)); src = d["p0"] if arm is None else d["p0"][arm]
    return {ast.literal_eval(k): v for k, v in src.items()}, d.get("shots",1024)

print("(A) ROTATION AXIS OF THE ROUND'S NOISE MAP -- is it Z-type?")
print("%-22s %8s %8s %8s %8s %10s" % ("dataset","angle","n_x","n_y","n_z","f_z=|n_z|"))
sets = [("rung2_result.json",None,"rung2 baseline"),
        ("rung6_result.json","0","rung6 plain"),
        ("rung6_result.json","1","rung6 echo"),
        ("rung5_result.json","0","rung5 plain"),
        ("rung5_result.json","1","rung5 DD")]
fzs = {}
for f, arm, lab in sets:
    p0, sh = load(f, arm)
    A, c = C.pair_from_p0(p0)
    ang, n = axis_angle(A)
    if n is None: print("%-22s  reflection" % lab); continue
    # bootstrap the axis
    keys=list(p0); base=np.array([p0[k] for k in keys]); fz=[]
    for _ in range(200):
        d=rng.binomial(1024, np.clip(base,0,1))/1024
        Ab,_=C.pair_from_p0(dict(zip(keys,d))); a2,n2=axis_angle(Ab)
        if n2 is not None: fz.append(abs(n2[2]))
    fzs[lab]=(abs(n[2]), np.std(fz))
    print("%-22s %8.2f %8.3f %8.3f %8.3f %6.3f +/-%.3f"
          % (lab, ang, n[0], n[1], n[2], abs(n[2]), np.std(fz)))

print("\n  f_z is the cosine of the angle between the rotation axis and Z.")
print("  W3-1's correction can only cancel the Z-projected part, so f_z is")
print("  exactly the parameter the power study is indexed by.")

print("\n(B) TEST STATISTICS FOR GATE G-VZ  (sigma=0.60 deg/arm @1024 shots)")
SIG, TOT, TH0 = 0.60, 6.67, 2.84
KS = np.array([0,-1,1,-2,2], float)
def truth(k, fz):
    tz=TOT*fz; tp=np.sqrt(max(TOT**2-tz**2,0))
    return np.sqrt(tp**2+(tz*(1+k))**2)
NB=20000
print("%5s | %-28s | %-28s | %-24s" % ("f_z",
      "V2 as registered (min<k0,3sig)","ASYM1  ang(+1)-ang(-1) > 0","ASYM2  ang(+2)-ang(-2) > 0"))
for fz in (0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98):
    yt = truth(KS,fz)
    Y = yt + rng.normal(0,SIG,(NB,5))
    v2  = np.mean((Y[:,0]-Y.min(axis=1)) >= 3*SIG*np.sqrt(2))
    a1  = np.mean((Y[:,2]-Y[:,1]) >= 3*SIG*np.sqrt(2))
    a2  = np.mean((Y[:,4]-Y[:,3]) >= 3*SIG*np.sqrt(2))
    e1  = (truth(1,fz)-truth(-1,fz))/(SIG*np.sqrt(2))
    e2  = (truth(2,fz)-truth(-2,fz))/(SIG*np.sqrt(2))
    print("%5.2f | power %4.0f%%   drop %5.2f deg   | power %4.0f%%  effect %5.1f sig | "
          "power %4.0f%%  effect %5.1f sig"
          % (fz,100*v2,truth(0,fz)-truth(-1,fz),100*a1,e1,100*a2,e2))
print("\n  Null for ASYM: with no signed Z response the +k and -k arms are")
print("  exchangeable, so E[ASYM]=0 exactly; drift enters both equally.")
