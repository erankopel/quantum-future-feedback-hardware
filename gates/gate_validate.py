#!/usr/bin/env python3
"""
GATE GCORE: reproduce the paper's own recorded hardware numbers from the
stored counts, using paper7_core. If these match, the analysis pipeline that
will process the NEW runs is trustworthy. Zero QPU.
"""
import json, ast, sys
import numpy as np
import paper7_core as C


def load_arms(fname, arm_keys):
    d = json.load(open(fname))
    out = {}
    for ak in arm_keys:
        p0 = {ast.literal_eval(k): v for k, v in d["p0"][ak].items()}
        out[ak] = p0
    return d, out


def approx(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


def main():
    ok = True

    # ---- rung6: plain / echo (the sharpest cross-check: eta, angle, index) --
    d6, arms6 = load_arms("rung6_result.json", ["0", "1"])
    rec = d6["arms"]
    for ak, name in (("0", "plain"), ("1", "echo")):
        got = C.analyze(arms6[ak])
        r = rec[name]
        checks = [
            ("eta",           got["eta"],            r["eta"],            2e-4),
            ("angle_deg",     got["angle_deg"],      r["angle_deg"],      0.05),
            ("eta_after_rot", got["eta_after_rot"],  r["eta_after_rot"],  2e-4),
            ("anisotropy",    got["anisotropy"],     r["anisotropy"],     2e-4),
        ]
        idx_ok = (got["index_measured"] == r["index_measured"] and
                  got["index_isotropic"] == r["index_isotropic"])
        line_ok = all(approx(g, e, t) for _, g, e, t in checks) and idx_ok
        ok = ok and line_ok
        print("rung6 %-5s  eta %.5f/%.5f  ang %.2f/%.2f  e' %.4f/%.4f  "
              "aniso %.4f/%.4f  idx %d/%d model %d/%d  [%s]"
              % (name, got["eta"], r["eta"], got["angle_deg"], r["angle_deg"],
                 got["eta_after_rot"], r["eta_after_rot"],
                 got["anisotropy"], r["anisotropy"],
                 got["index_measured"], r["index_measured"],
                 got["index_isotropic"], r["index_isotropic"],
                 "PASS" if line_ok else "FAIL"))

    # ---- rung5: plain / DD arms (eta + index) -------------------------------
    d5, arms5 = load_arms("rung5_result.json", ["0", "1"])
    rec5 = d5["arms"]
    for ak, name in (("0", "plain"), ("1", "DD")):
        got = C.analyze(arms5[ak])
        r = rec5[name]
        line_ok = (approx(got["eta"], r["eta"], 2e-4) and
                   got["index_measured"] == r["index"] and
                   got["index_isotropic"] == r["model_index"])
        ok = ok and line_ok
        print("rung5 %-5s  eta %.5f/%.5f  idx %d/%d model %d/%d  [%s]"
              % (name, got["eta"], r["eta"], got["index_measured"], r["index"],
                 got["index_isotropic"], r["model_index"],
                 "PASS" if line_ok else "FAIL"))

    # ---- rung4: the three delay blocks (eta + index + anisotropy) -----------
    d4 = json.load(open("rung4_result.json"))
    for bi, row in enumerate(d4["rows"]):
        p0 = {ast.literal_eval(k): v for k, v in d4["p0"][str(bi)].items()}
        got = C.analyze(p0)
        line_ok = (approx(got["eta"], row["eta"], 2e-4) and
                   got["index_measured"] == row["index"] and
                   got["index_isotropic"] == row["model_index"] and
                   approx(got["anisotropy"], row["anisotropy"], 2e-4))
        ok = ok and line_ok
        print("rung4 dt=%.1f  eta %.5f/%.5f  idx %d/%d model %d/%d  "
              "aniso %.4f/%.4f  [%s]"
              % (row["dt_us"], got["eta"], row["eta"], got["index_measured"],
                 row["index"], got["index_isotropic"], row["model_index"],
                 got["anisotropy"], row["anisotropy"],
                 "PASS" if line_ok else "FAIL"))

    # ---- rung2 baseline -----------------------------------------------------
    d2 = json.load(open("rung2_result.json"))
    p0 = {ast.literal_eval(k): v for k, v in d2["p0"].items()}
    got = C.analyze(p0)
    line_ok = (approx(got["eta"], d2["eta"], 2e-4) and
               got["index_measured"] == d2["implemented_index"] and
               got["index_isotropic"] == d2["model_index"] and
               approx(got["anisotropy"], d2["anisotropy"], 2e-4))
    ok = ok and line_ok
    print("rung2 base   eta %.5f/%.5f  idx %d/%d model %d/%d  aniso %.4f/%.4f  [%s]"
          % (got["eta"], d2["eta"], got["index_measured"], d2["implemented_index"],
             got["index_isotropic"], d2["model_index"], got["anisotropy"],
             d2["anisotropy"], "PASS" if line_ok else "FAIL"))

    print("\nGATE GCORE:", "PASS - analysis pipeline reproduces the series"
          if ok else "FAIL - do NOT trust new-data analysis")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
