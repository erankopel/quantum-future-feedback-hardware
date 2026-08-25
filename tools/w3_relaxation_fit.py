"""ZERO QPU. Fit the post-recalibration relaxation of the round's coherent
rotation. All points are the PLAIN arm: identical circuits, identical
transpile, identical seed, so every difference is the device."""
import json, os, numpy as np, datetime
from scipy.optimize import curve_fit

T0 = datetime.datetime(2026, 8, 25, 11, 40, 34, tzinfo=datetime.timezone.utc)
pts = [("W3-1  scan",   "2026-08-25T11:41:00Z",  7.83, 0.1288, 9),
       ("W3-1c rescan", "2026-08-25T11:53:30Z", 14.51, 0.1348, 11),
       ("W3-1b control","2026-08-25T12:05:00Z", 19.07, 0.1852, 8)]
led = json.load(open("rung8_ledger.json"))
for e in led["epochs"]:
    if e.get("fetched") and e.get("angle_deg") is not None:
        pts.append(("epoch %d" % e["epoch"], e["submitted_utc"],
                    e["angle_deg"], e["eta"], e["index_measured"]))

def mins(ts):
    ts = ts.replace("Z", "+00:00")
    return (datetime.datetime.fromisoformat(ts) - T0).total_seconds() / 60.0

t = np.array([mins(p[1]) for p in pts])
a = np.array([p[2] for p in pts]); et = np.array([p[3] for p in pts])

def relax(x, ainf, amp, tau):
    return ainf - amp * np.exp(-x / tau)

print("PLAIN-ARM RELAXATION, t measured from the 11:40:34 UTC recalibration")
print("  %-15s %7s %8s %8s %5s" % ("point", "t (min)", "angle", "eta", "idx"))
for p, tt in zip(pts, t):
    print("  %-15s %7.1f %8.2f %8.4f %5d" % (p[0], tt, p[2], p[3], p[4]))

for name, y, unit in (("angle", a, "deg"), ("eta", et, "")):
    try:
        p0 = [y.max(), y.max() - y.min(), 10.0]
        popt, pcov = curve_fit(relax, t, y, p0=p0, maxfev=20000)
        pe = np.sqrt(np.diag(pcov))
        res = y - relax(t, *popt)
        print("\n  %s(t) = %.3f - %.3f * exp(-t/%.2f)  %s" % (name, *popt, unit))
        print("     asymptote %.3f +/- %.3f | amplitude %.3f +/- %.3f | "
              "tau %.2f +/- %.2f min" % (popt[0], pe[0], popt[1], pe[1],
                                         popt[2], pe[2]))
        print("     rms residual %.3f %s over %d points" % (np.sqrt((res**2).mean()), unit, len(t)))
    except Exception as ex:
        print("\n  %s fit failed: %s" % (name, str(ex)[:100]))

# corrected-arm summary
print("\nCORRECTED ARM (theta = +8.52 deg, fixed, measured 11:49-11:58)")
print("  %-9s %7s %8s %8s %8s %10s" % ("epoch","t(min)","plain","corr","drop","idx p->c"))
for e in led["epochs"]:
    c = e.get("corrected")
    if c:
        print("  %-9d %7.1f %8.2f %8.2f %8.2f %6d -> %d"
              % (e["epoch"], mins(e["submitted_utc"]), e["angle_deg"],
                 c["angle_deg"], c["angle_deg"] - e["angle_deg"],
                 e["index_measured"], c["index_measured"]))
