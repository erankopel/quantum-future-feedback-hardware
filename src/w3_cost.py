#!/usr/bin/env python3
"""Single source of truth for the window-3 QPU price.

Amendment 1 (25 Aug, pre-run) fitted cost = N*(a + b*shots) on 19 historical
charges. Amendment 1a (25 Aug, after W3-0b): the ping charged 5 s against 3.36 s
predicted, an apparent 1.49x. It is NOT a price rise. The ping is by far the
smallest job ever run in this series (12 circuits; previous smallest 157), and a
fixed PER-JOB overhead invisible on a 200 s job is a third of a 3 s one. Re-fit
on all 20 charges with a per-job intercept:

    cost = n_jobs * c0 + N * (a + b * shots),   c0 = 2.30 s

  model             max under-prediction   rms      ping
  no intercept          +32.8 %           1.16 s    3.36 vs 5
  per-job intercept      +1.8 %           0.76 s    5.55 vs 5

The intercept model is adopted; the safety factor drops to 1.02 because it
covers every one of the 20 charges. IMPORTANT: cost() now takes n_jobs, because
a design split into chunks pays the intercept once per chunk. W3-2's eight
chunks carry 8 * 2.30 = 18.4 s of overhead that the old model priced at zero.
"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
A_DEF, B_DEF, C0_DEF, SAFETY = 0.005748, 2.676436e-4, 0.0, 1.05


def params():
    f = os.path.join(HERE, "cost_model_w3.json")
    if os.path.isfile(f):
        d = json.load(open(f))
        return (float(d["a_per_circuit"]), float(d["b_per_shot"]),
                float(d.get("c0_per_job", 0.0)),
                float(d.get("safety_factor", SAFETY)), "W3-0b ping + 19 charges")
    return A_DEF, B_DEF, C0_DEF, SAFETY, "19 historical charges"


def cost(n, shots, loaded=True, n_jobs=1):
    a, b, c0, safe, _ = params()
    return (n_jobs * c0 + n * (a + b * shots)) * (safe if loaded else 1.0)


def source():
    return params()[4]
