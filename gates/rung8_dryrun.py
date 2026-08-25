"""W3-3 DRY RUN harness: exercise rung8_epoch.submit_epoch's build path with
the sampler neutralised, so nothing can reach the QPU."""
import os, sys, json, math
sys.path.insert(0,".")
import rung8_epoch as R8
import submit_core as S

# --- neutralise the sampler: any attempt to run raises loudly -------------
class Blocked:
    def __init__(self,*a,**k): pass
    def run(self,*a,**k): raise SystemExit("DRY RUN: sampler.run() blocked")
R8.SamplerV2 = Blocked

from qiskit import transpile
backend = S.service().backend("ibm_kingston")
print("backend:", backend.name, "| operational:", backend.status().operational,
      "| queue:", backend.status().pending_jobs)

# --- calibration snapshot / triple health (this is W3-0) ------------------
snap = R8.cal_snapshot(backend)
print("\n--- W3-0 CALIBRATION SNAPSHOT / TRIPLE HEALTH ---")
for q,v in snap["qubits"].items():
    print("  q%-3s  T1=%7.1f us  T2=%7.1f us  readout_err=%.5f"
          % (q, v["t1_us"] or -1, v["t2_us"] or -1, v["readout_err"]))
print("  two-qubit gate:", snap["twoq_gate"])
for k,v in snap["links_on_M"].items():
    print("  link %-10s err=%.5f" % (k, v))
print("  last_update_date:", snap["last_update_date"])
need = {frozenset((21,22)), frozenset((21,36))}
have = {frozenset(eval(k)) for k in snap["links_on_M"]}
print("  BOTH SERIES LINKS PRESENT:", "YES" if need <= have else "NO  *** STOP ***")

# --- build both arms exactly as submit_epoch would -----------------------
circuits, keys = [], []
for st in S.all_settings():
    circuits.append(S.build_tomo(*st)); keys.append(list(st))
tq = transpile(circuits, backend=backend, optimization_level=3,
               initial_layout=list(S.TRIPLE), seed_transpiler=S.SEED)
print("\n--- W3-3 BUILD ---")
print("  plain arm: %d circuits, width %d, max depth %d"
      % (len(tq), tq[0].num_qubits, max(t.depth() for t in tq)))
THETA_TEST = -2.84
corr = R8._corrected(tq, THETA_TEST)
from collections import Counter
ok = True
for a,b in zip(tq, corr):
    ca, cb = Counter(dict(a.count_ops())), Counter(dict(b.count_ops()))
    if cb["rz"] - ca["rz"] != 15: ok = False; print("   rz delta", cb["rz"]-ca["rz"])
    for op in (set(ca)|set(cb)) - {"rz"}:
        if ca[op] != cb[op]: ok = False; print("   op mismatch", op)
print("  corrected arm: +15 rz per circuit, all other ops identical:",
      "PASS" if ok else "FAIL")

# --- budget guard behaviour ----------------------------------------------
u = S.service().usage()
print("\n--- BUDGET ---")
print("  live consumed %ss / limit %ss (cap in code %ss)"
      % (u["usage_consumed_seconds"], u["usage_limit_seconds"], R8.CAP_TOTAL))
for arm,nn in (("plain",72),("both",144)):
    c = nn*(R8.A_PC + R8.B_PS*R8.SHOTS)
    print("  epoch %-6s %3d circuits x %d shots -> %.1f s" % (arm,nn,R8.SHOTS,c))

# --- theta plumbing ------------------------------------------------------
print("\n--- THETA PLUMBING ---")
print("  w3_theta.json present:", os.path.isfile(R8.THETA_FILE),
      "-> arm alternation would start:", 
      "both" if os.path.isfile(R8.THETA_FILE) else "plain (correct pre-scan)")

# --- prove the submit path is blocked ------------------------------------
print("\n--- SUBMIT PATH BLOCKED CHECK ---")
try:
    R8.submit_epoch(dry=True)
    print("   submit_epoch(dry=True) returned without touching the sampler")
    R8.submit_epoch()
    print("  *** WARNING: live submit_epoch returned without hitting the blocker")
except SystemExit as e:
    print("  ", e, "<- the live path is the only one that reaches the sampler")
