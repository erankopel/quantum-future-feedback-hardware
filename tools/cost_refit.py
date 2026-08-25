import numpy as np, json
# (label, n_circuits, shots, charged_seconds, account)
D = [
 ("rung2  da5vbee1", 72,4096, 80,"A1"), ("rung3  da5vjmbo",72,4096, 80,"A1"),
 ("rung4  da60aue1",216,4096,239,"A1"), ("rung5  da60ejbo",144,4096,158,"A1"),
 ("rung6  da615rjo",144,1024, 42,"A1"), ("rung7  da66vt48",288,1024, 81,"A2"),
 ("rung9  da6701kg",436,1024,118,"A2"),
 ("r10-2122 c1",250,512,36,"A2"),("r10-2122 c2",250,512,36,"A2"),
 ("r10-2122 c3",250,512,36,"A2"),("r10-2122 c4",250,512,36,"A2"),
 ("r10-2122 c5",250,512,36,"A2"),("r10-2122 c6",157,512,23,"A2"),
 ("r10-2136 c1",250,480,34,"A2"),("r10-2136 c2",250,480,34,"A2"),
 ("r10-2136 c3",250,480,34,"A2"),("r10-2136 c4",250,480,34,"A2"),
 ("r10-2136 c5",250,480,34,"A2"),("r10-2136 c6",157,480,22,"A2"),
]
n  = np.array([d[1] for d in D], float)
sh = np.array([d[2] for d in D], float)
y  = np.array([d[3] for d in D], float)

a_old, b_old = 0.0121, 2.577e-4
def pred(a,b): return n*(a+b*sh)

# OLS refit of cost = n*a + n*shots*b
X = np.vstack([n, n*sh]).T
(a_new, b_new), *_ = np.linalg.lstsq(X, y, rcond=None)

# conservative: smallest scale k such that k*model >= every charge
r_old = y/pred(a_old,b_old); r_new = y/pred(a_new,b_new)

print("PRE-REGISTERED MODEL  a=%.6f  b=%.6e" % (a_old,b_old))
print("OLS REFIT (19 charges) a=%.6f  b=%.6e" % (a_new,b_new))
print()
print("%-18s %5s %5s %7s %9s %8s %9s %8s" %
      ("job","n","shots","charged","pre-reg","err%","refit","err%"))
for i,d in enumerate(D):
    p0,p1 = pred(a_old,b_old)[i], pred(a_new,b_new)[i]
    print("%-18s %5d %5d %7.0f %9.1f %+7.1f %9.1f %+7.1f" %
          (d[0],d[1],d[2],y[i],p0,100*(p0-y[i])/y[i],p1,100*(p1-y[i])/y[i]))
print()
print("pre-reg : max under-prediction %.1f%%   mean %+.1f%%   worst ratio charged/pred = %.4f"
      % (100*(max(r_old)-1), 100*(np.mean(1/r_old)-1), max(r_old)))
print("refit   : max under-prediction %.1f%%   mean %+.1f%%   worst ratio charged/pred = %.4f"
      % (100*(max(r_new)-1), 100*(np.mean(1/r_new)-1), max(r_new)))
SAFE = float(np.ceil(max(r_new)*100)/100)
print("\nRECOMMENDED SAFETY FACTOR on refit model: %.2f  (covers every observed charge)" % SAFE)

print("\n--- WINDOW-3 LEDGER, three models ---")
jobs=[("W3-0b cost ping", 12,1024),("W3-1 VZ scan (5 arm)",360,1024),
      ("W3-1b VZ control 2122",216,1024),
      ("W3-2 deep GST L<=8",1813,384),("W3-3 epoch plain",72,1024),
      ("W3-3 epoch both",144,1024)]
print("%-24s %6s %6s %9s %9s %11s" % ("item","n","shots","pre-reg","refit","refit*%.2f"%SAFE))
for lab,nn,ss in jobs:
    c0=nn*(a_old+b_old*ss); c1=nn*(a_new+b_new*ss)
    print("%-24s %6d %6d %9.1f %9.1f %11.1f" % (lab,nn,ss,c0,c1,c1*SAFE))
json.dump(dict(a_per_circuit=a_new,b_per_shot=b_new,safety_factor=SAFE,
               n_charges=len(D),source="19 actual charges, accounts A1+A2, ibm_kingston"),
          open("cost_model_v2.json","w"), indent=1)
print("\nwrote cost_model_v2.json")
