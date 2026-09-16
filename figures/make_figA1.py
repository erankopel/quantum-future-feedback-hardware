#!/usr/bin/env python3
"""
Paper A, figure 1 (two panels), regenerated from the deposited JSONs of the
Paper VII repository (data/sensitivity_register2_f64.json,
data/affine_register4.json, data/rung2_result.json).

A: n_idx(eta) for the five operating points of the round unitary, labelled by
   their noiseless index, with the measured device point and the isotropic
   model at the same eta.
B: certified lower bound on the index against shots per setting for the
   simulated channel with implemented index 46: independent per-entry balls
   against the 72-symbol affine propagation.

Every plotted endpoint is asserted against the JSONs (same discipline as the
deposited figure script). Vector PDF for the journal, PNG for preview.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# data beside the script, or in ../data when run from figures/ in the repo
DATA = HERE if os.path.exists(os.path.join(HERE, "rung2_result.json")) \
    else os.path.join(HERE, "..", "data")
OUTDIR = os.path.join(HERE, "..", "tex") if os.path.isdir(os.path.join(HERE, "..", "tex")) else HERE

S = json.load(open(os.path.join(DATA, "sensitivity_register2_f64.json")))
A4 = json.load(open(os.path.join(DATA, "affine_register4.json")))
HW = json.load(open(os.path.join(DATA, "rung2_result.json")))

# operating points in the order of the paper's Table 1, keyed by the JSON tags
ORDER = ["SCAN4-TR1 floor28", "PROBE60-TR1 floor60", "QC1 floor81",
         "QR1 floor89", "D3 floor109"]
COL = {"SCAN4-TR1 floor28": "#0072B2", "PROBE60-TR1 floor60": "#E69F00",
       "QC1 floor81": "#009E73", "QR1 floor89": "#56B4E9",
       "D3 floor109": "#CC79A7"}
NAME = {"SCAN4-TR1 floor28": "P1", "PROBE60-TR1 floor60": "P2",
        "QC1 floor81": "P3", "QR1 floor89": "P4", "D3 floor109": "P5"}

t = S["targets"]
assert t["QR1 floor89"]["n_ideal"] == 88
assert t["D3 floor109"]["n_ideal"] == 107
assert HW["backend"] == "ibm_kingston" and HW["shots"] == 4096
assert HW["implemented_index"] == 11 and HW["model_index"] == 10
assert abs(HW["eta"] - 0.10022) < 5e-4
rows = [r for r in A4["rows"] if r["eps"] == 1e-3]
by_shots = {r["shots"]: r for r in rows}
assert by_shots[10**6]["ball_lower"] == 19 and by_shots[10**6]["affine_lower"] == 34
assert by_shots[10**9]["affine_lower"] == 45
assert rows[0]["implemented"] == 46

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "legend.fontsize": 8.5, "pdf.fonttype": 42})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.1))
for ax in (axA, axB):
    ax.grid(True, which="major", color="0.88", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------------- A: the index against the contraction
for tag in ORDER:
    d = t[tag]
    e = np.array([p[0] for p in d["curve"]])
    n = np.array([p[1] for p in d["curve"]], float)
    m = n > 0
    axA.step(e[m], n[m], where="post", color=COL[tag], lw=1.6)
    axA.text(1.15e-5, d["n_ideal"], f"{NAME[tag]}: $n(0)={d['n_ideal']}$",
             fontsize=7.5, color=COL[tag], va="center", ha="left",
             bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.4))
eta_hw = HW["eta"]
axA.plot([eta_hw], [HW["implemented_index"]], marker="*", ms=13,
         color="#14191D", mec="white", mew=0.6, ls="none", zorder=7)
axA.annotate("ibm_kingston, P4\nmeasured pair: 11\nisotropic model, same $\\eta$: 10",
             xy=(eta_hw * 0.90, HW["implemented_index"] - 0.3),
             xytext=(1.3e-4, 8), ha="left", va="center", fontsize=7.5, color="0.15",
             bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.3),
             arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8, shrinkB=3))
axA.set_xscale("log")
axA.set_xlim(1e-5, 0.35)
axA.set_ylim(0, 115)
axA.set_xlabel(r"contraction $\eta$ of the round's affine pair")
axA.set_ylabel(r"index $n_{\mathrm{idx}}$ of the contracted pair")
axA.set_title("(a) five operating points of Eq. (1)", loc="left")

# ---------------- B: the propagation gain
shots = sorted(by_shots)
ball = [by_shots[s]["ball_lower"] for s in shots]
aff = [by_shots[s]["affine_lower"] for s in shots]
axB.axhline(46, color="0.25", lw=1.0, ls=(0, (4, 3)))
axB.text(1.25e5, 47.0, "implemented index 46", fontsize=8, color="0.25")
axB.plot(shots, ball, "o-", color="#0072B2", lw=1.6, ms=5,
         label="independent per-entry balls")
axB.plot(shots, aff, "s-", color="#E69F00", lw=1.6, ms=5,
         label="72-symbol affine propagation")
for s, b, a in zip(shots, ball, aff):
    axB.annotate("", xy=(s, a - 0.6), xytext=(s, b + 0.6),
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=0.8))
axB.text(10**6, (19 + 34) / 2, " +15", fontsize=8.5, color="0.3", va="center")
axB.set_xscale("log")
axB.set_xlim(6e4, 2e9)
axB.set_ylim(0, 52)
axB.set_xlabel(r"shots per setting (joint confidence $1-10^{-3}$)")
axB.set_ylabel("certified lower bound on the index")
axB.set_title(r"(b) certified bound against shots, $n_{\mathrm{idx}}=46$", loc="left")
axB.legend(loc="lower right", frameon=False)

fig.tight_layout(w_pad=2.0)
fig.savefig(os.path.join(OUTDIR, "figA1_ruler.pdf"))
fig.savefig(os.path.join(OUTDIR, "figA1_ruler.png"), dpi=200)
print("wrote figA1_ruler.pdf/.png")
