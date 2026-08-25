"""The W3-0b ping charged 5 s against 3.36 s predicted, a naive scale of 1.489.
Applying that scale uniformly would price W3-2 at 308 s and end Path B. But the
ping is by far the SMALLEST job in the series (12 circuits; the previous
smallest was 157), so a fixed PER-JOB overhead that is invisible on a 200 s job
is 30-40 per cent of a 3 s one. Test that hypothesis against all 20 charges."""
import numpy as np, json
D = [(72,4096,80),(72,4096,80),(216,4096,239),(144,4096,158),(144,1024,42),
     (288,1024,81),(436,1024,118),
     (250,512,36),(250,512,36),(250,512,36),(250,512,36),(250,512,36),(157,512,23),
     (250,480,34),(250,480,34),(250,480,34),(250,480,34),(250,480,34),(157,480,22),
     (12,1024,5)]                                   # <- the W3-0b ping
n  = np.array([d[0] for d in D], float)
sh = np.array([d[1] for d in D], float)
y  = np.array([d[2] for d in D], float)

# model 1: no intercept (what the plan used)
X1 = np.vstack([n, n*sh]).T
(a1,b1),*_ = np.linalg.lstsq(X1, y, rcond=None)
# model 2: with a per-JOB intercept
X2 = np.vstack([np.ones_like(n), n, n*sh]).T
(c0,a2,b2),*_ = np.linalg.lstsq(X2, y, rcond=None)

def rep(name,p):
    r = y - p
    print("%-26s max under-pred %+6.2f s (%+5.1f%%)  rms %.2f s  ping %.2f vs 5"
          % (name, r.max(), 100*(r/y).max(), np.sqrt((r**2).mean()), p[-1]))
p1 = n*(a1+b1*sh); p2 = c0 + n*(a2+b2*sh)
print("MODEL 1  no intercept   a=%.6f b=%.6e" % (a1,b1)); rep("  ",p1)
print("MODEL 2  per-job c0     c0=%.3f s a=%.6f b=%.6e" % (c0,a2,b2)); rep("  ",p2)
print()
print("%5s %6s %8s %9s %9s" % ("n","shots","charged","no-icpt","with-icpt"))
for i,d in enumerate(D):
    mark = "  <- W3-0b ping" if i==len(D)-1 else ""
    print("%5d %6d %8.0f %9.2f %9.2f%s" % (d[0],d[1],d[2],p1[i],p2[i],mark))

SAFE = float(np.ceil(max(y/p2)*100)/100)
print("\nsafety factor on MODEL 2: %.2f" % SAFE)
print("\nRE-PRICED LEDGER (model 2, jobs counted, safety %.2f):" % SAFE)
def cost(njobs,n_,sh_): return (njobs*c0 + n_*(a2+b2*sh_))*SAFE
items=[("W3-1 scan       1 job",1,360,1024),("W3-1b control   1 job",1,216,1024),
       ("W3-2 GST        8 jobs",8,1813,384),("epoch plain     1 job",1,72,1024),
       ("epoch both      1 job",1,144,1024)]
for l,j,n_,s_ in items: print("  %-24s %6.1f s" % (l,cost(j,n_,s_)))
A = 5 + cost(1,360,1024)+cost(1,216,1024)+5*cost(1,144,1024)+5*cost(1,72,1024)
B = 5 + cost(1,360,1024)+cost(8,1813,384)
print("  PATH A  scan+control+10 epochs = %.1f s, reserve %.1f s" % (A,600-A))
print("  PATH B  scan+deep GST          = %.1f s, then %d plain epochs fit in the 550 cap"
      % (B, int((550-B)//cost(1,72,1024))))
json.dump(dict(c0_per_job=c0,a_per_circuit=a2,b_per_shot=b2,safety_factor=SAFE,
               n_charges=len(D),note="per-job intercept; validated on the W3-0b ping"),
          open("cost_model_w3.json","w"), indent=1)
print("\nrewrote cost_model_w3.json")
