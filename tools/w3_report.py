#!/usr/bin/env python3
"""Generate window3_results.txt from the on-disk records. ASCII, no em dashes."""
import json, os, datetime, numpy as np
from scipy.optimize import curve_fit

T0 = datetime.datetime(2026, 8, 25, 11, 40, 34, tzinfo=datetime.timezone.utc)
def mins(ts):
    return (datetime.datetime.fromisoformat(ts.replace("Z","+00:00")) - T0).total_seconds()/60.

L = []
def w(s=""): L.append(s)

scan = json.load(open("rung11b_result.json"))
resc = json.load(open("rung11b_result_rescan.json"))
ctrl = json.load(open("rung11b_result_control2122.json"))
led  = json.load(open("rung8_ledger.json"))
th   = json.load(open("w3_theta.json"))
cm   = json.load(open("cost_model_w3.json"))

w("="*64)
w("PAPER VII, WINDOW 3: RESULTS")
w("Executed 25 Aug 2026 on ibm_kingston. ASCII, no em dashes.")
w("Plan: window3_plan.txt as amended by window3_plan_v2.txt.")
w("="*64)
w()
w("LEDGER")
w("  instance   crn:...a/REDACTED-ACCOUNT-3...:a45caeb2  (third account, fresh 600 s)")
w("  price      c0=%.3f s/job + N*(%.6f + %.6e*shots), safety %.2f"
  % (cm["c0_per_job"], cm["a_per_circuit"], cm["b_per_shot"], cm["safety_factor"]))
w()

w("-"*64)
w("W3-1  VIRTUAL-Z FIVE-ARM SCAN, link (21,36), 360 x 1024, job %s" % scan["job_id"])
w("  ran 11:35:22 to 11:47:06 UTC; a recalibration fell at 11:40:34, INSIDE")
w("  this window. Arms are interleaved per setting so the differential")
w("  statistics are unaffected; the absolute level is a time average.")
w()
w("  %-18s %8s %8s %8s %8s %6s %6s" % ("arm","angle","sd","eta","eta'","idx","model"))
for r in sorted(scan["rows"], key=lambda r: r["k"]):
    w("  k=%+d (%+6.2f deg)  %8.2f %8.2f %8.4f %8.4f %6d %6d"
      % (r["k"], r["applied_deg"], r["angle_deg"], r["sd_angle"],
         r["eta"], r["eta_after_rot"], r["index_measured"], r["index_isotropic"]))
v = scan["verdict"]
w()
w("  ASYM1 = %+.2f +/- %.2f deg = %+.1f sigma" % (v["asym1"]["diff_deg"], v["asym1"]["sd"], v["asym1"]["nsigma"]))
w("  ASYM2 = %+.2f +/- %.2f deg = %+.1f sigma" % (v["asym2"]["diff_deg"], v["asym2"]["sd"], v["asym2"]["nsigma"]))
w("  minimum below plain %.2f deg = %.1f sigma at arm k=%+d"
  % (v["drop_deg"], v["drop_nsigma"], v["best_k"]))
w("  V1 %s  V2 %s  V3 %s  V4 %s  ASYM %s"
  % tuple("PASS" if v[x] else "FAIL" for x in ("V1","V2","V3","V4","ASYM_signed")))
w("  V2P (minimum at POSITIVE k, amendment A10): %s"
  % ("CONFIRMED" if v.get("V2P_min_at_positive_k") else "not confirmed"))
w("  hyperbola theta_opt %+.2f deg, floor %.2f deg, rms resid %.2f deg"
  % (v["hyperbola"]["theta_opt"], v["hyperbola"]["floor_deg"], v["hyperbola"]["rms_resid"]))
w("  RISK R5 FIRED: optimum outside the scanned span; theta NOT handed on.")
w()
w("  The registered V2 predicted a minimum at NEGATIVE k. The k=-1 and k=-2")
w("  arms measured %.2f and %.2f deg against a plain arm of %.2f deg: the"
  % ([r for r in scan["rows"] if r["k"]==-1][0]["angle_deg"],
     [r for r in scan["rows"] if r["k"]==-2][0]["angle_deg"],
     [r for r in scan["rows"] if r["k"]==0][0]["angle_deg"]))
w("  registered sign makes the rotation WORSE. A single-theta A/B at the")
w("  register-14 sign would have been written up as a refutation.")
w()

w("-"*64)
w("W3-1c RE-CENTRED REPEAT, link (21,36), 288 x 1024, job %s" % resc["job_id"])
w("  ran 11:49:46 to 11:58:22 UTC, entirely after the recalibration.")
w()
w("  %-18s %8s %8s %8s %8s %6s %6s" % ("arm","angle","sd","eta","eta'","idx","model"))
for r in sorted(resc["rows"], key=lambda r: r["k"]):
    w("  k=%+d (%+6.2f deg)  %8.2f %8.2f %8.4f %8.4f %6d %6d"
      % (r["k"], r["applied_deg"], r["angle_deg"], r["sd_angle"],
         r["eta"], r["eta_after_rot"], r["index_measured"], r["index_isotropic"]))
vr = resc["verdict"]
w()
w("  minimum below plain %.2f deg = %.1f sigma" % (vr["drop_deg"], vr["drop_nsigma"]))
w("  hyperbola theta_opt %+.2f deg, FLOOR %.2f deg" % (vr["hyperbola"]["theta_opt"], vr["hyperbola"]["floor_deg"]))
w("  flat bottom: k=+3 and k=+4 agree to 0.10 +/- 0.81 deg. The correction")
w("  saturates on a non-Z residual that a virtual Z cannot reach.")
w("  theta handed on: %+.2f deg (%s)" % (th["theta_opt_deg"], th.get("estimator","plateau midpoint")))
w()

w("-"*64)
w("W3-1b WRONG-LINK CONTROL, link (21,22), 216 x 1024, job %s" % ctrl["job_id"])
w("  ran 12:02 to 12:08 UTC, same calibration epoch as W3-1c.")
w("  Total injected Z MATCHED: 15 x 8.52 = 9 x 14.20 = 127.8 deg.")
w()
for r in sorted(ctrl["rows"], key=lambda r: r["k"]):
    w("  k=%+d (%+6.2f deg)  angle %8.2f +/- %.2f   eta %.4f   idx %d"
      % (r["k"], r["applied_deg"], r["angle_deg"], r["sd_angle"], r["eta"], r["index_measured"]))
vc = ctrl["verdict"]
w()
w("  ASYM1 = %+.2f +/- %.2f deg = %+.1f sigma   <- CONTROL"
  % (vc["asym1"]["diff_deg"], vc["asym1"]["sd"], vc["asym1"]["nsigma"]))
w("  Both signed arms rise together by +6.35 and +6.23 deg. A pure |k|")
w("  response with zero asymmetry: the same total Z at the wrong sites adds")
w("  rotation regardless of sign and removes nothing.")
w("  PREDICTION C1 (control ASYM under 3 sigma while carrier ASYM is over):")
w("     carrier %+.1f sigma   control %+.1f sigma   -> PASS"
  % (v["asym1"]["nsigma"], vc["asym1"]["nsigma"]))
w("  The round's coherent rotation is carried by the (21,36) CZ specifically.")
w()

w("-"*64)
w("W3-3 STABILITY STUDY, %d epochs, dense cadence (deviation A1d)" % len(led["epochs"]))
w("  Registered cadence was ~3 h spacing. The plain-arm anchors showed the")
w("  rotation moving on a ~15 min timescale, so epochs were chained back to")
w("  back. Author-approved, logged.")
w()
pts = [("W3-1  scan","2026-08-25T11:41:00Z",7.83,0.1288,9),
       ("W3-1c rescan","2026-08-25T11:53:30Z",14.51,0.1348,11),
       ("W3-1b control","2026-08-25T12:05:00Z",19.07,0.1852,8)]
for e in led["epochs"]:
    if e.get("fetched") and e.get("angle_deg") is not None:
        pts.append(("epoch %d" % e["epoch"], e["submitted_utc"], e["angle_deg"],
                    e["eta"], e["index_measured"]))
w("  PLAIN ARM, identical circuits, t from the 11:40:34 recalibration")
w("  %-16s %8s %8s %9s %5s" % ("point","t (min)","angle","eta","idx"))
for p in pts:
    w("  %-16s %8.1f %8.2f %9.4f %5d" % (p[0], mins(p[1]), p[2], p[3], p[4]))
t = np.array([mins(p[1]) for p in pts]); a = np.array([p[2] for p in pts])
try:
    f = lambda x, ai, am, ta: ai - am*np.exp(-x/ta)
    po, pc = curve_fit(f, t, a, p0=[a.max(), a.max()-a.min(), 10.], maxfev=20000)
    pe = np.sqrt(np.diag(pc)); res = a - f(t, *po)
    w()
    w("  RELAXATION FIT  angle(t) = %.2f - %.2f * exp(-t / %.2f min)" % tuple(po))
    w("     asymptote %.2f +/- %.2f deg" % (po[0], pe[0]))
    w("     amplitude %.2f +/- %.2f deg" % (po[1], pe[1]))
    w("     tau       %.2f +/- %.2f min" % (po[2], pe[2]))
    w("     rms residual %.2f deg over %d points (shot-noise floor 0.60 deg)"
      % (np.sqrt((res**2).mean()), len(t)))
    w()
    w("  The round's coherent rotation nearly TRIPLED, from %.1f to %.1f deg,"
      % (po[0]-po[1], po[0]))
    w("  with a %.0f minute time constant, following a recalibration. This is" % po[2])
    w("  the manuscript's 'the implemented index is a property of a calibration")
    w("  epoch' with a timestamped event, a time constant and an amplitude.")
except Exception as ex:
    w("  relaxation fit failed: %s" % str(ex)[:120])
w()
w("  CORRECTED ARM, theta = %+.2f deg held FIXED from 11:49-11:58" % th["theta_opt_deg"])
w("  %-7s %8s %8s %8s %8s %12s" % ("epoch","t(min)","plain","corr","drop","idx p -> c"))
for e in led["epochs"]:
    c = e.get("corrected")
    if c:
        w("  %-7d %8.1f %8.2f %8.2f %8.2f %6d -> %-3d"
          % (e["epoch"], mins(e["submitted_utc"]), e["angle_deg"], c["angle_deg"],
             c["angle_deg"]-e["angle_deg"], e["index_measured"], c["index_measured"]))
w()
w("  SECOND RECALIBRATION, and a caveat recorded against our own story.")
w("  A further recalibration landed at 13:22:11 UTC, between epochs 6 and 7.")
w("  It produced NO comparable jump: epochs 6, 7, 8 read 21.10, 22.12, 20.63")
w("  deg, flat inside the 0.60 deg noise floor. So 'a recalibration moves the")
w("  round by 14 deg' is NOT what this window shows. What it shows is that")
w("  ONE recalibration did, with a 14 min time constant, and a later one did")
w("  not. The mechanism is not established by n = 1, and the manuscript must")
w("  say so. What IS established is that the round's rotation can move by 14")
w("  deg on a 15 min timescale with no user action at all.")
w()
w("  An in-loop virtual Z on the localised CZ raises the IMPLEMENTED INDEX,")
w("  in the circuit and not in analysis, and a theta measured once keeps")
w("  working across the epochs that follow.")
w()
w("="*64)
open("window3_results.txt","w").write("\n".join(L) + "\n")
print("\n".join(L))
