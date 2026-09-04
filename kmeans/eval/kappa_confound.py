"""Is the inertia spread across kappa signal, or run-to-run noise?

Two spreads, same data, same fixed iteration count:
  A. vary kappa, hold the initial centroids fixed   -> the "signal" I sought
  B. hold kappa fixed, vary the initial centroids   -> the noise floor
If B is as large as A, A measured nothing.
"""
import numpy as np, torch, time
from mp_kmeans.clustering import KMeansPlusPlus
D="/global/homes/j/jbellav/m4646/hoon/mpkmeans-dsets/data/data_yandex.bin"
d,k,ITERS = 200,64,20
X = np.fromfile(D, dtype=np.float32).reshape(-1,d)
rng = np.random.default_rng(1)
X = np.ascontiguousarray(X[np.sort(rng.choice(X.shape[0],150000,replace=False))])
n = X.shape[0]
Xt = torch.from_numpy(X).cuda().contiguous()

def centroids(seed):
    r = np.random.default_rng(seed)
    return torch.from_numpy(np.ascontiguousarray(X[r.choice(n,k,replace=False)])).cuda().contiguous()

def run(kernel, kappa, C0):
    m = KMeansPlusPlus(n_clusters=k, kernel=kernel, kappa=kappa, max_iter=ITERS,
                       tol=1e-45, normalize=None, random_state=1, init_method="random")
    m.fit(Xt, C0.clone())
    return float((((Xt.double()-m.cluster_centers_.double()[m.labels_])**2).sum()).item())

C_fixed = centroids(1)
run("fp32",5.0,C_fixed)   # warm

print("A. vary kappa, SAME initial centroids (seed 1)")
A=[]
for kap in (0.0,0.5,1.0,2.0,3.0,5.0,10.0,50.0,1e6):
    I = run("fp16_fp32",kap,C_fixed); A.append(I)
    print(f"   kappa={kap:<8.4g} inertia={I:.10e}")
ref = run("fp32",5.0,C_fixed)
print(f"   {'fp32':14s} inertia={ref:.10e}")
sA = (max(A)-min(A))/ref
print(f"   spread across kappa            : {sA:.3e} relative")

print("\nB. kappa FIXED at 5, vary the initial centroids (seeds 1..8)")
B=[]
for s in range(1,9):
    C = centroids(s)
    I = run("fp16_fp32",5.0,C); B.append(I)
    print(f"   seed={s}          inertia={I:.10e}")
sB = (max(B)-min(B))/np.mean(B)
print(f"   spread across seeds            : {sB:.3e} relative")
print(f"\n   noise / signal = {sB/sA:.0f}x")
