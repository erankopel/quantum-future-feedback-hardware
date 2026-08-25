"""ZERO QPU. Two things the pre-registered plan needs but does not yet have:
   (1) the shot-noise sigma on the polar rotation angle, so GATE G-VZ's
       ">= 3 sigma" criterion is actually computable;
   (2) the link census of the compiled round (how many CZ on each link),
       which sets the control-arm design.
"""
import json, ast, numpy as np
import paper7_core as C

rng = np.random.default_rng(11)

def boot_sigma(p0, shots, nboot=400):
    keys = list(p0)
    base = np.array([p0[k] for k in keys])
    ang, eta, etap, idx = [], [], [], []
    for _ in range(nboot):
        draw = rng.binomial(shots, np.clip(base,0,1)) / shots
        b = C.analyze(dict(zip(keys, draw)))
        if b["angle_deg"] is not None:
            ang.append(b["angle_deg"]); eta.append(b["eta"])
            etap.append(b["eta_after_rot"]); idx.append(b["index_measured"])
    return (np.std(ang), np.std(eta), np.std(etap),
            np.mean(ang), len(set(idx)), min(idx), max(idx))

print("SHOT-NOISE FLOOR on the explorer bundle (parametric bootstrap, 400 draws)")
print("%-26s %6s | %8s %8s | %8s %8s %8s | %s" %
      ("dataset","shots","angle","sd(ang)","eta","sd(eta)","sd(eta')","idx range"))
rows=[]
for fname,label,armkey in (("rung2_result.json","rung2 baseline",None),
                           ("rung6_result.json","rung6 plain","0"),
                           ("rung6_result.json","rung6 echo","1")):
    d = json.load(open(fname))
    src = d["p0"] if armkey is None else d["p0"][armkey]
    p0 = {ast.literal_eval(k): v for k, v in src.items()}
    native = d["shots"] if "shots" in d else 1024
    for shots in (native, 1024):
        sd_a, sd_e, sd_ep, m_a, nidx, imin, imax = boot_sigma(p0, shots)
        print("%-26s %6d | %8.2f %8.2f | %8.4f %8.4f %8.4f | %d..%d" %
              (label, shots, m_a, sd_a, C.analyze(p0)["eta"], sd_e, sd_ep, imin, imax))
        rows.append((label,shots,m_a,sd_a))
        if shots==native and native==1024: break

print("\nG-VZ FEASIBILITY at 1024 shots:")
for label,shots,m,sd in rows:
    if shots==1024:
        print("  %-16s angle %5.2f deg, 1 sigma = %.2f deg -> 3 sigma needs a "
              "%.2f deg drop (%.0f%% of the angle)" % (label,m,sd,3*sd,100*3*sd/m))

# ---- link census of the compiled round ----------------------------------
from qiskit import transpile
import submit_core as S
from collections import Counter
backend = S.service().backend("ibm_kingston")
base = [S.build_tomo(*st) for st in S.all_settings()]
tq = transpile(base, backend=backend, optimization_level=3,
               initial_layout=list(S.TRIPLE), seed_transpiler=S.SEED)
cen = Counter()
for inst in tq[0].data:
    if inst.operation.name in ("cz","ecr","cx"):
        cen[tuple(sorted(tq[0].find_bit(q).index for q in inst.qubits))] += 1
print("\nCOMPILED ROUND, two-qubit gate census (circuit 0 of 72):")
for k,v in sorted(cen.items()):
    print("   link %-10s : %2d gates  %s" % (k, v,
          "<- localised carrier" if k==(21,36) else "<- control link" if k==(21,22) else ""))
print("   total 2Q gates:", sum(cen.values()))
print("   depth:", tq[0].depth(), " width:", tq[0].num_qubits)
