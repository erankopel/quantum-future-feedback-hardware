"""ZERO QPU. Unit tests for the amended G-VZ gate: every case asserts BOTH the
path and the write/refuse decision, and each is run over 8 noise seeds so a
result cannot pass by luck of the seed (which is how the first self-test
missed the grid-edge bug)."""
import numpy as np, json, ast
from scipy.linalg import polar
import paper7_core as C, w3_vfit as V

RIN={"z+":[0,0,1.],"z-":[0,0,-1.],"x+":[1.,0,0],"x-":[-1.,0,0],"y+":[0,1.,0],"y-":[0,-1.,0]}
MS={"x":0,"y":1,"z":2}; KS=(0,-1,1,-2,2); TH0=2.84; SHOTS=1024

d2=json.load(open("rung2_result.json"))
p0={ast.literal_eval(k):v for k,v in d2["p0"].items()}
A0,c0=C.pair_from_p0(p0)
N=A0@np.linalg.inv(C.Aid); O,_=polar(N)
ang=np.degrees(np.arccos(np.clip((O.trace()-1)/2,-1,1)))
w=np.array([O[2,1]-O[1,2],O[0,2]-O[2,0],O[1,0]-O[0,1]]); w/=np.linalg.norm(w)
DZ=ang*abs(w[2])

def rz(deg):
    t=np.radians(deg);ct,st=np.cos(t),np.sin(t)
    return np.array([[ct,-st,0],[st,ct,0],[0,0,1.]])

def synth(A,c,rng,shots=SHOTS,scale=1.0):
    out={}
    for prep in C.PREP:
        r=(A@np.array(RIN[prep])+c)*scale
        for ms in C.BASIS:
            p=(1+float(np.clip(r[MS[ms]],-1,1)))/2
            for f in (0,1):
                for l in (0,1): out[(prep,ms,f,l)]=rng.binomial(shots,p)/shots
    return out

def case(name, delta_per_k, want_path, want_write, drift=0.0):
    # drift: an EXTRA angle reduction proportional to |k|, i.e. a response that
    # depends on the magnitude of the insertion and not its sign. That is the
    # textbook artefact the plan's "both signed arms move the same way" branch
    # exists to catch.
    got=[]
    for seed in range(8):
        rng=np.random.default_rng(500+seed); rows=[]
        for k in KS:
            Ak=rz(k*delta_per_k - drift*abs(k))@A0
            b=V.bundle_with_sigma(synth(Ak,c0,rng),SHOTS,nboot=150,seed=900+seed)
            b["k"]=k; b["applied_deg"]=k*TH0; rows.append(b)
        v=V.gvz_verdict(rows,TH0)
        got.append((v["path"],v["write_theta"],v.get("theta_opt_deg")))
    paths=[g[0] for g in got]; writes=[g[1] for g in got]
    th=[g[2] for g in got if g[2] is not None]
    okp=all(p==want_path for p in paths)
    okw=all(bool(x)==want_write for x in writes)
    print("%-38s path %-22s write %-18s theta %s  [%s]"
          %(name,"/".join(sorted(set(paths))),
            "/".join(sorted({str(x) for x in writes})),
            ("%+.2f..%+.2f"%(min(th),max(th))) if th else "n/a",
            "PASS" if (okp and okw) else "*** FAIL ***"))
    return okp and okw

ok=True
ok&=case("(a) true effect, +k cancels",      DZ,   "A", True)
ok&=case("(d) reversed transfer, -k cancels",-DZ,  "A", True)
ok&=case("(b) null, correction inert",        0.0, "B", False)
ok&=case("(c) half step: optimum at grid edge",DZ/2,"A", False)
DRIFT="DRIFT ARTEFACT -> 3-arm repeat before deciding"
ok&=case("(e1) |k| response, both arms UP",   0.0, DRIFT, False, drift= DZ*0.55)
ok&=case("(e2) |k| response, both arms DOWN", 0.0, DRIFT, False, drift=-DZ*0.55)
print("\nG-VZ GATE UNIT TESTS:", "ALL PASS" if ok else "*** FAILURES ABOVE ***")
