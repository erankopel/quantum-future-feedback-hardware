"""ZERO QPU. End-to-end proof that the W3-1 analysis chain returns the RIGHT
verdict before the run: synthesise the five arms from the real rung2 Bloch
pair, push them through w3_vfit exactly as rung11b analyze will, and check the
gate under (a) a true localised Z error, (b) the null, (c) a drift artefact."""
import json, ast, numpy as np
from scipy.linalg import polar
import paper7_core as C, w3_vfit as V

rng = np.random.default_rng(23)
RIN = {"z+":np.array([0,0,1.]),"z-":np.array([0,0,-1.]),
       "x+":np.array([1.,0,0]),"x-":np.array([-1.,0,0]),
       "y+":np.array([0,1.,0]),"y-":np.array([0,-1.,0])}
MS = {"x":0,"y":1,"z":2}

def p0_from_pair(A,c,shots=None):
    out={}
    for prep in C.PREP:
        r = A@RIN[prep]+c
        for ms in C.BASIS:
            e = float(np.clip(r[MS[ms]],-1,1)); p=(1+e)/2
            for f in (0,1):
                for l in (0,1):
                    v = rng.binomial(shots,p)/shots if shots else p
                    out[(prep,ms,f,l)] = v
    return out

def rz(deg):
    t=np.radians(deg); ct,st=np.cos(t),np.sin(t)
    return np.array([[ct,-st,0],[st,ct,0],[0,0,1.]])

d2 = json.load(open("rung2_result.json"))
p0 = {ast.literal_eval(k):v for k,v in d2["p0"].items()}
A0,c0 = C.pair_from_p0(p0)
Ar,cr = C.pair_from_p0(p0_from_pair(A0,c0))
print("forward/inverse round-trip max err: %.2e" % np.abs(Ar-A0).max())
N=A0@np.linalg.inv(C.Aid); O,_=polar(N)
ang=np.degrees(np.arccos(np.clip((O.trace()-1)/2,-1,1)))
w=np.array([O[2,1]-O[1,2],O[0,2]-O[2,0],O[1,0]-O[0,1]]); w/=np.linalg.norm(w)
DZ = ang*abs(w[2])
print("rung2 real map: angle %.2f deg, axis n_z %.3f -> Z-part %.2f deg\n"%(ang,w[2],DZ))

KS=(0,-1,1,-2,2); TH0=2.84; SHOTS=1024
def run(label, delta_per_k, expect):
    rows=[]
    for k in KS:
        Ak = rz(k*delta_per_k)@A0
        b = V.bundle_with_sigma(p0_from_pair(Ak,c0,SHOTS), SHOTS, nboot=200)
        b["k"]=k; b["applied_deg"]=k*TH0; rows.append(b)
    v=V.gvz_verdict(rows,TH0)
    print("%-34s" % label)
    print("   arms:", "  ".join("k%+d:%5.2f"%(r["k"],r["angle_deg"]) for r in sorted(rows,key=lambda r:r["k"])))
    print("   sigma %.2f | hyp theta_opt %s | ASYM1 %s | V2 %s | PATH %s | write_theta %s"
          % (v["sigma_bar_deg"],
             "%+.2f"%v["hyperbola"]["theta_opt"] if v["hyperbola"] else "n/a",
             "%+.1f sig"%v["asym1"]["nsigma"] if v["asym1"] else "n/a",
             "PASS" if v["V2"] else "fail", v["path"],
             v["write_theta"]))
    print("   expected: %s   -> %s\n" % (expect,
          "OK" if expect.split()[0]==v["path"] else "*** MISMATCH ***"))
    return v

# (a) true localised Z error: k=-1 cancels it exactly
run("(a) TRUE effect, +k cancels", DZ, "A  theta_opt near +2.84")
# (b) null: the correction does nothing (wrong link / no Z error)
run("(b) NULL, correction inert", 0.0, "B  no signed response")
run("(d) REVERSED transfer, -k cancels", -DZ, "A  theta_opt near -2.84")
# (c) half-size effect (register-14 number 2x too big)
run("(c) HALF-size per-gate phase", DZ/2, "A  optimum at grid edge k=+2")
