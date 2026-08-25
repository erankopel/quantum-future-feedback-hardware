#!/usr/bin/env python3
"""Paper VII, register 15 figures. Vector PDF for REVTeX (PRA two-column).
Palette: Okabe-Ito subset #0072B2 / #D55E00 / #009E73, validated colourblind-safe;
marker shape and line style carry the same identity so the panels survive
greyscale printing."""
import json, datetime, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

BLUE, VERM, GREEN, INK, MUT = "#0072B2", "#D55E00", "#009E73", "#1a1a1a", "#7a7a7a"
COL1, COL2 = 3.375, 7.0          # PRA single- and double-column widths (inch)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1, "legend.frameon": False,
    "axes.edgecolor": "#555555", "axes.labelcolor": INK,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "text.color": INK, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42})

def recess(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, lw=0.4, color="#e2e2e2", zorder=0)
    ax.set_axisbelow(True)

scan = json.load(open("rung11b_result.json"))
resc = json.load(open("rung11b_result_rescan.json"))
ctrl = json.load(open("rung11b_result_control2122.json"))
led  = json.load(open("rung8_ledger.json"))
def arms(d): return sorted(d["rows"], key=lambda r: r["applied_deg"])

def hyp(x, p, q, x0): return np.sqrt(p**2 + (q*(x - x0))**2)
def fithyp(x, y):
    f = lambda pp: hyp(x, *pp) - y
    s = least_squares(f, [max(y.min(),.1), (y.max()-y.min())/(x.max()-x.min()+1e-9)+.1,
                          x[int(np.argmin(y))]], max_nfev=8000)
    return s.x

# ================= FIGURE 2: the scan and the control ====================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL2, 2.5))

r = arms(scan); x = np.array([a["applied_deg"] for a in r]); y = np.array([a["angle_deg"] for a in r])
e = np.array([a["sd_angle"] for a in r])
r2 = arms(resc); x2 = np.array([a["applied_deg"] for a in r2]); y2 = np.array([a["angle_deg"] for a in r2])
e2 = np.array([a["sd_angle"] for a in r2])
xs  = np.linspace(x.min()-0.8, x.max()+0.8, 400)     # never extrapolate past the arms
xs2 = np.linspace(x2.min()-0.8, x2.max()+0.8, 400)
axA.plot(xs, hyp(xs, *fithyp(x, y)), color=BLUE, lw=1.0, zorder=2)
axA.errorbar(x, y, yerr=e, fmt="o", ms=4.5, color=BLUE, mfc=BLUE, mec="white",
             mew=0.7, ecolor=BLUE, elinewidth=0.9, capsize=2, zorder=3,
             label="scan 1 (spans recalibration)")
axA.plot(xs2, hyp(xs2, *fithyp(x2, y2)), color=VERM, lw=1.0, ls="--", zorder=2)
axA.errorbar(x2, y2, yerr=e2, fmt="s", ms=4.2, color=VERM, mfc="white", mec=VERM,
             mew=1.1, ecolor=VERM, elinewidth=0.9, capsize=2, zorder=3,
             label="repeat (after it)")
axA.axvline(0, color=MUT, lw=0.6, ls=":", zorder=1)
axA.annotate("registered V2\npredicted the\nminimum here", xy=(-2.84, 10.97),
             xytext=(-7.4, 3.2), fontsize=6.5, color=MUT, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=MUT, lw=0.6,
                             connectionstyle="arc3,rad=-0.25"))
axA.set_xlabel(r"applied correction per CZ, $k\,\theta_0$  (deg)")
axA.set_ylabel(r"noise-map rotation angle  (deg)")
axA.set_title(r"(a)  carrier link $(21,36)$, 15 insertions", loc="left", pad=6)
axA.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95,
           edgecolor="#dcdcdc", borderpad=0.5).get_frame().set_linewidth(0.5)
recess(axA); axA.set_ylim(0, 21); axA.set_xlim(-8.2, 13.2)

rc = arms(ctrl); xc = np.array([a["applied_deg"] for a in rc])
yc = np.array([a["angle_deg"] for a in rc]); ec = np.array([a["sd_angle"] for a in rc])
axB.errorbar(xc, yc, yerr=ec, fmt="^", ms=5, color=GREEN, mfc=GREEN, mec="white",
             mew=0.7, ecolor=GREEN, elinewidth=0.9, capsize=2, zorder=3)
axB.plot(xc, yc, color=GREEN, lw=0.9, ls="-.", zorder=2, alpha=0.55)  # guide to the eye
axB.axhline(yc[1], color=MUT, lw=0.6, ls=":", zorder=1)
axB.axvline(0, color=MUT, lw=0.6, ls=":", zorder=1)
a1 = ctrl["verdict"]["asym1"]
axB.text(0.5, 0.06, r"$\mathrm{ASYM}_1 = %+.2f \pm %.2f$ deg $= %+.1f\,\sigma$"
         % (a1["diff_deg"], a1["sd"], a1["nsigma"]), transform=axB.transAxes,
         ha="center", fontsize=7, color=INK)
axB.set_xlabel(r"applied correction per CZ  (deg)")
axB.set_ylabel(r"noise-map rotation angle  (deg)")
axB.set_title(r"(b)  control link $(21,22)$, matched total $Z$", loc="left", pad=6)
axB.text(0.5, 0.90, "both signs rise together", transform=axB.transAxes,
         ha="center", fontsize=7, color=MUT)
recess(axB); axB.set_ylim(0, 29); axB.set_xlim(-17, 17)
fig.tight_layout(w_pad=2.0)
fig.savefig("fig2_vzscan.pdf"); fig.savefig("fig2_vzscan.png", dpi=220)
print("fig2_vzscan written")

# ================= FIGURE 3: the calibration clock =======================
T0 = datetime.datetime(2026, 8, 25, 11, 40, 34, tzinfo=datetime.timezone.utc)
def mins(ts): return (datetime.datetime.fromisoformat(ts.replace("Z","+00:00"))-T0).total_seconds()/60.
pts = [("2026-08-25T11:41:00Z", 7.83, 0.1288), ("2026-08-25T11:53:30Z", 14.51, 0.1348),
       ("2026-08-25T12:05:00Z", 19.07, 0.1852)]
for ep in led["epochs"]:
    if ep.get("fetched") and ep.get("angle_deg") is not None:
        pts.append((ep["submitted_utc"], ep["angle_deg"], ep["eta"]))
t = np.array([mins(p[0]) for p in pts]); ang = np.array([p[1] for p in pts])
et = np.array([p[2] for p in pts])
rel = lambda x, ai, am, ta: ai - am*np.exp(-x/ta)
po, pc = curve_fit(rel, t, ang, p0=[ang.max(), np.ptp(ang), 10.], maxfev=20000)
pe = np.sqrt(np.diag(pc))

fig, ax = plt.subplots(figsize=(COL1, 2.35))
ts = np.linspace(-4, 112, 400)
ax.axvline(0, color=VERM, lw=0.9, ls="--", zorder=1)
ax.text(4.0, 6.1, "recalibration\n11:40:34 UTC", fontsize=6.5, color=VERM, va="bottom")
ax.plot(ts, rel(ts, *po), color=BLUE, lw=1.1, zorder=2)
ax.errorbar(t, ang, yerr=0.62, fmt="o", ms=4.2, color=BLUE, mfc=BLUE, mec="white",
            mew=0.7, ecolor=BLUE, elinewidth=0.9, capsize=2, zorder=3)
ax.axhline(po[0], color=MUT, lw=0.6, ls=":", zorder=1)
ax.text(0.97, 0.955, r"$\theta(t)=%.1f-%.1f\,e^{-t/\tau}$, $\tau=%.1f\pm%.1f$ min"
        % (po[0], po[1], po[2], pe[2]), transform=ax.transAxes, ha="right",
        fontsize=6.8, color=INK)
ax.set_xlabel("time since recalibration  (min)")
ax.set_ylabel(r"rotation angle, plain arm  (deg)")
recess(ax); ax.set_ylim(4, 25); ax.set_xlim(-5, 112)
fig.savefig("fig3_relaxation.pdf"); fig.savefig("fig3_relaxation.png", dpi=220)
print("fig3_relaxation written  tau=%.2f+/-%.2f  asym=%.2f  amp=%.2f" % (po[2],pe[2],po[0],po[1]))

# ================= FIGURE 4: index recovery + decomposition ==============
c_gap = np.mean([(a["eta"]-a["eta_after_rot"])/a["angle_deg"]**2 for a in arms(scan)])
eps = [e for e in led["epochs"] if e.get("corrected")]
idx = np.arange(len(eps)); lab = ["%d" % e["epoch"] for e in eps]
ip = [e["index_measured"] for e in eps]; ic = [e["corrected"]["index_measured"] for e in eps]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL1, 3.5), sharex=True)
wd = 0.34
ax1.bar(idx-wd/2-0.01, ip, wd, color=BLUE, label="plain", zorder=3)
ax1.bar(idx+wd/2+0.01, ic, wd, color=VERM, label="corrected", zorder=3, hatch="///",
        edgecolor="white", lw=0.0)
for i,(a,b) in enumerate(zip(ip,ic)):
    ax1.text(i-wd/2-0.01, a+0.15, str(a), ha="center", fontsize=6.5, color=INK)
    ax1.text(i+wd/2+0.01, b+0.15, str(b), ha="center", fontsize=6.5, color=INK)
ax1.set_ylabel("implemented index")
ax1.set_title(r"(a)  in-loop correction, $\theta=+8.52$ deg held fixed", loc="left", pad=5)
ax1.legend(loc="upper right", ncol=2, frameon=True, facecolor="white",
           framealpha=0.95, edgecolor="#dcdcdc").get_frame().set_linewidth(0.5)
recess(ax1); ax1.set_ylim(0, 16.5)

d_obs = np.array([e["eta"]-e["corrected"]["eta"] for e in eps])
d_rot = np.array([c_gap*(e["angle_deg"]**2 - e["corrected"]["angle_deg"]**2) for e in eps])
ax2.bar(idx, d_rot, 0.55, color=GREEN, label="rotation removed", zorder=3)
ax2.bar(idx, d_obs-d_rot, 0.55, bottom=d_rot+0.0016, color=VERM, zorder=3,
        label="genuine decay removed", hatch="///", edgecolor="white", lw=0.0)
for i,(o,rr) in enumerate(zip(d_obs,d_rot)):
    ax2.text(i, rr/2, "%.0f%%" % (100*rr/o), ha="center", va="center",
             fontsize=6.5, color="white")
    ax2.text(i, rr+(o-rr)/2, "%.0f%%" % (100*(o-rr)/o), ha="center", va="center",
             fontsize=6.5, color="white")
ax2.set_ylabel(r"fall in $\eta$")
ax2.set_xlabel("epoch"); ax2.set_xticks(idx); ax2.set_xticklabels(lab)
ax2.set_title(r"(b)  decomposition of the fall in $\eta$", loc="left", pad=5)
ax2.legend(loc="upper right", ncol=1, frameon=True, facecolor="white",
           framealpha=0.95, edgecolor="#dcdcdc").get_frame().set_linewidth(0.5)
recess(ax2); ax2.set_ylim(0, 0.163)
fig.tight_layout(h_pad=1.4)
fig.savefig("fig4_correction.pdf"); fig.savefig("fig4_correction.png", dpi=220)
print("fig4_correction written  c=%.3e  rot frac %.0f%%" % (c_gap, 100*np.mean(d_rot/d_obs)))
