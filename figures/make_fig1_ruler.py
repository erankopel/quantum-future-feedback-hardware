#!/usr/bin/env python3
"""
Paper VII figure 1: the ruler and the propagation gain. Two panels.

A: n_idx(eta) for the five certified targets (register 2 curves), log-x, with
   the band of current two-qubit link errors shaded (best to median Eagle ecr,
   eta = 11.507 eps).
B: certified lower bound on the implemented index versus shots per setting at
   eps = 1e-3 (implemented 46): independent per-entry balls (register 3
   machinery) against the 72-symbol affine propagation (register 4).

Assertion gates: every plotted endpoint is asserted against the shipped JSONs,
so the figure cannot silently drift from the registers.

Palette: Okabe-Ito subset, validated (dataviz six-checks; the one 6-8 CVD pair
is covered by direct labels on every line).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tex", "fig1_ruler.png")

S = json.load(open(os.path.join(HERE, "sensitivity_register2_f64.json")))
A4 = json.load(open(os.path.join(HERE, "affine_register4.json")))
HW = json.load(open(os.path.join(HERE, "rung2_result.json")))
ETA_PER_EPS = S["eta_per_eps"]

COL = {"SCAN4-TR1 floor28": "#0072B2", "PROBE60-TR1 floor60": "#E69F00",
       "QC1 floor81": "#009E73", "QR1 floor89": "#56B4E9",
       "D3 floor109": "#CC79A7"}
LAB = {"SCAN4-TR1 floor28": "floor 28", "PROBE60-TR1 floor60": "floor 60",
       "QC1 floor81": "QC1", "QR1 floor89": "QR1", "D3 floor109": "D3"}

# ---- gates on the register-2 curves
t = S["targets"]
assert t["QR1 floor89"]["n_ideal"] == 88
assert t["D3 floor109"]["n_ideal"] == 107 and t["D3 floor109"]["readout_offset"] == 1
assert t["QR1 floor89"]["reference_indices"]["1.00e-03"] == 46

# ---- gates on the hardware run (register 5)
assert HW["backend"] == "ibm_kingston" and HW["shots"] == 4096
assert HW["implemented_index"] == 11 and HW["model_index"] == 10
assert abs(HW["eta"] - 0.10022) < 5e-4

# ---- gates on the register-4 rows (eps = 1e-3)
rows = [r for r in A4["rows"] if r["eps"] == 1e-3]
by_shots = {r["shots"]: r for r in rows}
assert by_shots[10**6]["ball_lower"] == 19 and by_shots[10**6]["affine_lower"] == 34
assert by_shots[10**9]["affine_lower"] == 45
assert rows[0]["implemented"] == 46

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 4.0))
for ax in (axA, axB):
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------------- A: the ruler
eta_lo, eta_hi = ETA_PER_EPS * 3.47e-3, ETA_PER_EPS * 7.79e-3
axA.axvspan(eta_lo, eta_hi, color="0.90", lw=0, zorder=0)
axA.text(np.sqrt(eta_lo * eta_hi), 100, "current links\n(best to median)",
         ha="center", fontsize=8, color="0.35")
ends = []
for tag, d in t.items():
    e = np.array([p[0] for p in d["curve"]])
    n = np.array([p[1] for p in d["curve"]], float)
    m = n > 0
    axA.step(e[m], n[m], where="post", color=COL[tag], lw=1.8)
    ends.append((tag, d["n_ideal"]))
    axA.text(1.15e-5, d["n_ideal"], LAB[tag], fontsize=8.5, color=COL[tag],
             va="center", ha="left",
             bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
# the measured device (register 5): one-integer miss made visible
eta_hw = HW["eta"]
axA.plot([eta_hw], [HW["model_index"]], marker="o", ms=7, mfc="white",
         mec="0.25", mew=1.4, ls="none", zorder=6)
axA.plot([eta_hw], [HW["implemented_index"]], marker="*", ms=15,
         color="#14191D", mec="white", mew=0.7, ls="none", zorder=7)
axA.annotate("ibm_kingston, measured: 11",
             xy=(eta_hw * 1.06, HW["implemented_index"] + 0.6),
             xytext=(0.33, 33), ha="right", fontsize=8.5, color="0.15",
             arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8,
                             shrinkB=2))
axA.annotate("isotropic model, same $\\eta$: 10",
             xy=(eta_hw * 1.06, HW["model_index"] - 0.6),
             xytext=(0.33, 24), ha="right", fontsize=8.5, color="0.35",
             arrowprops=dict(arrowstyle="-", color="0.7", lw=0.8,
                             shrinkB=2))

axA.set_xscale("log")
axA.set_xlim(1e-5, 0.35)
axA.set_ylim(0, 115)
axA.set_xlabel(r"contraction $\eta$   ($\eta = 11.507\,\epsilon_{2q}$)")
axA.set_ylabel(r"index of the implemented channel  $n_{\mathrm{idx}}(\eta)$")
axA.set_title("A. Five certified targets as fidelity rulers", loc="left",
              fontsize=10.5)

# ---------------- B: the propagation gain at eps = 1e-3
shots = sorted(by_shots)
ball = [by_shots[s]["ball_lower"] for s in shots]
aff = [by_shots[s]["affine_lower"] for s in shots]
axB.axhline(46, color="0.25", lw=1.0, ls=(0, (4, 3)))
axB.text(1.25e5, 46.9, "implemented index 46", fontsize=8.5, color="0.25")
axB.plot(shots, ball, "o-", color="#0072B2", lw=1.8, ms=6,
         label="independent balls (reg. 3)")
axB.plot(shots, aff, "s-", color="#E69F00", lw=1.8, ms=6,
         label="72-symbol affine (reg. 4)")
for s, b, a in zip(shots, ball, aff):
    axB.annotate("", xy=(s, a - 0.6), xytext=(s, b + 0.6),
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
axB.text(10**6, (19 + 34) / 2, " +15", fontsize=9, color="0.3", va="center")
axB.set_xscale("log")
axB.set_xlim(6e4, 2e9)
axB.set_ylim(0, 52)
axB.set_xlabel(r"shots per setting  (joint confidence $1-10^{-3}$)")
axB.set_ylabel("certified lower bound on the index")
axB.set_title("B. What the correlated propagation buys "
              r"($\epsilon_{2q}=10^{-3}$)", loc="left", fontsize=10.5)
axB.legend(loc="lower right", fontsize=8.5, frameon=False)

fig.tight_layout(w_pad=2.4)
fig.savefig(OUT, dpi=600)
print("wrote", OUT)
