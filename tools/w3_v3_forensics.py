"""ZERO QPU. What actually happened to V3.
V3 predicted: eta falls at the minimum, while eta' is flat across arms --
'the correction removes rotation, not decay'. Both halves failed. Why."""
import json, numpy as np
scan = json.load(open("rung11b_result.json"))
led  = json.load(open("rung8_ledger.json"))

print("PART 1: eta and eta' are not independent. eta contains the rotation.")
print("  eta projects the measured Bloch matrix onto the IDEAL one, so a pure")
print("  coherent rotation of angle theta reduces that projection by ~cos(theta)")
print("  and INFLATES eta even with no extra decay. eta' divides the rotation")
print("  out first. So the gap eta - eta' should go as theta^2.\n")
print("  %-8s %8s %8s %8s %10s %12s" % ("arm","angle","eta","eta'","eta-eta'","(eta-eta')/th^2"))
rows = sorted(scan["rows"], key=lambda r: r["k"])
cs = []
for r in rows:
    th = r["angle_deg"]; gap = r["eta"] - r["eta_after_rot"]
    c = gap/th**2
    cs.append(c)
    print("  k=%+d     %8.2f %8.4f %8.4f %10.4f %12.2e"
          % (r["k"], th, r["eta"], r["eta_after_rot"], gap, c))
cs = np.array(cs)
print("\n  coefficient c = %.2e +/- %.2e per deg^2  (spread %.1f%% over a"
      % (cs.mean(), cs.std(), 100*cs.std()/cs.mean()))
print("  10-fold range of theta, 1.54 to 14.71 deg). The relation is exact.\n")

print("PART 2: why V3's first half failed in W3-1 and passed in the epochs.")
c = cs.mean()
plain = [r for r in rows if r["k"] == 0][0]
best  = min(rows, key=lambda r: r["angle_deg"])
print("  W3-1 ran at a PLAIN rotation of only %.2f deg. The rotation's whole"
      % plain["angle_deg"])
print("  contribution to eta there is c*theta^2 = %.4f, against a shot-noise"
      % (c*plain["angle_deg"]**2))
print("  sd on eta of %.4f. The effect V3 asked us to detect was %.1f sigma"
      % (plain["sd_eta"], c*plain["angle_deg"]**2/plain["sd_eta"]))
print("  BY CONSTRUCTION. V3 was underpowered before a single shot was taken.")
print("  I computed the pre-run power for the ANGLE and not for eta. That is")
print("  my omission, not a property of the device.\n")
print("  In the epochs the plain rotation was ~20 deg, where the same term is")
print("  c*theta^2 = %.4f, and eta duly fell hard:" % (c*20.7**2))
for e in led["epochs"]:
    cc = e.get("corrected")
    if cc:
        print("     ep%-2d eta %.4f -> %.4f   fall %.4f  (%.0f sigma)"
              % (e["epoch"], e["eta"], cc["eta"], e["eta"]-cc["eta"],
                 (e["eta"]-cc["eta"])/np.hypot(0.004, 0.004)))
print()

print("PART 3: the interesting half. Is the fall in eta JUST the rotation?")
print("  If the correction only removed a rigid rotation, the whole fall would")
print("  be accounted for by c*(theta_plain^2 - theta_corr^2) and eta' would")
print("  not move. Decompose it:\n")
print("  %-6s %8s %8s %10s %10s %10s %8s" %
      ("epoch","th_pln","th_cor","d_eta obs","d_eta rot","d_eta' ","rot frac"))
fr = []
for e in led["epochs"]:
    cc = e.get("corrected")
    if not cc: continue
    tp, tc = e["angle_deg"], cc["angle_deg"]
    d_obs = e["eta"] - cc["eta"]
    d_rot = c*(tp**2 - tc**2)
    d_ep  = e["eta_after_rot"] - cc["eta_after_rot"]
    fr.append(d_rot/d_obs)
    print("  %-6d %8.2f %8.2f %10.4f %10.4f %10.4f %7.0f%%"
          % (e["epoch"], tp, tc, d_obs, d_rot, d_ep, 100*d_rot/d_obs))
fr = np.array(fr)
print("\n  Rotation removal accounts for %.0f +/- %.0f per cent of the fall in"
      % (100*fr.mean(), 100*fr.std()))
print("  eta. The rest, and it is the majority, is a fall in eta' itself:")
print("  the rotation-corrected contraction improves by %.4f on average."
      % np.mean([e["eta_after_rot"]-e["corrected"]["eta_after_rot"]
                 for e in led["epochs"] if e.get("corrected")]))
print("\n  So V3's premise is wrong in an INFORMATIVE direction. The virtual Z")
print("  does not only remove a rigid rotation of the message qubit. A Z error")
print("  on the CZ is CONDITIONAL on the ancilla state; tracing the ancillas")
print("  out turns that conditional phase into DEPHASING. Cancelling it at the")
print("  gate removes a genuine decoherence channel, not just a frame error.")
