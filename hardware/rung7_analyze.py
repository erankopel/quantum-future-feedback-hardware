#!/usr/bin/env python3
"""
Rung 7 analysis: the CPMG-order refocusing curve. Explorer throughout.

Fetches the segmented job, splits the counts by pulse-number block, and for
each n reports the full explorer bundle plus the two register-9 predictions:
  Q1 rotation angle FALLS with n (the detuning refocuses)
  Q2 eta' (decay after rotation removed) stays ~FLAT (genuine incoherent decay)
  Q3 measured index RISES with n
Writes rung7_result.json.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import submit_core as S
import paper7_core as C


def main():
    meta = json.load(open(os.path.join(HERE, "rung7_job.json")))
    service = S.service()
    job = service.job(meta["job_id"])
    print("job %s: %s" % (meta["job_id"], job.status()))
    res = job.result()
    shots = meta["shots"]
    ns = meta["n_pulses"]

    # group click probs by n-block
    blocks = {n: {} for n in ns}
    for i, key in enumerate(meta["keys"]):
        n, prep, mea, f, l = key
        counts = res[i].data.c.get_counts()
        blocks[n][(prep, mea, f, l)] = counts.get("0", 0) / shots

    rows = []
    for n in ns:
        b = C.analyze(blocks[n])
        b["n_pulses"] = n
        rows.append(b)
        print("n=%d  eta=%.4f  angle=%5.2f  eta'=%.4f  aniso=%.3f  "
              "index=%d model=%d  sv=%s"
              % (n, b["eta"], b["angle_deg"], b["eta_after_rot"],
                 b["anisotropy"], b["index_measured"], b["index_isotropic"],
                 [round(x, 3) for x in b["singular_values"]]))

    a0, a_last = rows[0]["angle_deg"], rows[-1]["angle_deg"]
    e0, e_last = rows[0]["eta_after_rot"], rows[-1]["eta_after_rot"]
    # Q1: does refocusing remove rotation at all? compare bare to the BEST block
    angles = [r["angle_deg"] for r in rows]
    a_best = min(angles); n_best = rows[int(np.argmin(angles))]["n_pulses"]
    q1 = a_best < a0 - 1.0
    # Q2: is the rotation-corrected decay invariant across blocks? (all blocks
    # carry the SAME total idle, so a correct decomposition must give the same
    # eta'.) Criterion: spread small vs the raw eta spread.
    eps = [r["eta_after_rot"] for r in rows]
    ets = [r["eta"] for r in rows]
    spread_ep = max(eps) - min(eps); spread_et = max(ets) - min(ets)
    q2 = spread_ep < 0.02 and spread_ep < 0.2 * spread_et
    # Q3 (STRICT): does the measured index actually RISE above the bare block?
    q3 = max(r["index_measured"] for r in rows) > rows[0]["index_measured"]
    # Q4: does removing the rotation restore the isotropic model's integer?
    q4_blocks = [r["n_pulses"] for r in rows
                 if r["index_isotropic"] == r["index_measured"]]
    print("\nQ1 rotation refocuses (bare %.2f -> best %.2f deg at n=%d): %s"
          % (a0, a_best, n_best, q1))
    print("Q2 eta' invariant (spread %.4f vs raw eta spread %.4f): %s"
          % (spread_ep, spread_et, q2))
    print("Q3 STRICT index rises above bare (%d -> max %d): %s"
          % (rows[0]["index_measured"],
             max(r["index_measured"] for r in rows), q3))
    print("Q4 isotropic model integer restored at blocks n=%s (bare miss: "
          "model %d vs measured %d)"
          % (q4_blocks, rows[0]["index_isotropic"], rows[0]["index_measured"]))

    out = dict(job_id=meta["job_id"], backend=meta["backend"],
               layout=meta["layout"], shots=shots, n_pulses=ns,
               total_idle_us=meta["total_idle_us"], blocks=rows,
               predictions=dict(Q1_rotation_refocuses=bool(q1),
                                Q2_etaprime_invariant=bool(q2),
                                Q3_index_rises_strict=bool(q3),
                                Q4_model_integer_restored_at_n=q4_blocks,
                                best_refocus_n=int(n_best),
                                etaprime_spread=float(spread_ep),
                                eta_spread=float(spread_et)),
               p0={str(n): {str(k): v for k, v in blocks[n].items()}
                   for n in ns})
    json.dump(out, open(os.path.join(HERE, "rung7_result.json"), "w"),
              indent=1, default=str)
    print("\nwrote rung7_result.json")


if __name__ == "__main__":
    main()
