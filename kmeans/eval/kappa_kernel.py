"""Kernel accuracy vs kappa, with the clustering dynamics taken out.

The sweep over full fits mixes two things: how accurate the distance kernel is,
and where a 20-iteration run happens to land.  This fixes one set of centroids
and scores the kernel's output directly against an FP64 ground truth, so only
the first is left.
"""
import sys, numpy as np, torch
from mp_kmeans import euclidean_cuda as E
from mp_kmeans.clustering import KMeansPlusPlus

path, d, k, n_sub = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
X = np.fromfile(path, dtype=np.float32).reshape(-1, d)
rng = np.random.default_rng(1)
if 0 < n_sub < X.shape[0]:
    X = np.ascontiguousarray(X[np.sort(rng.choice(X.shape[0], n_sub, replace=False))])
n = X.shape[0]
Xt = torch.from_numpy(X).cuda().contiguous()
C0 = torch.from_numpy(np.ascontiguousarray(X[rng.choice(n, k, replace=False)])).cuda().contiguous()

# use CONVERGED centroids: that is the regime the test actually runs in, where
# many points sit near-equidistant from two centres and cancellation bites
m = KMeansPlusPlus(n_clusters=k, kernel="fp32", kappa=5.0, max_iter=50, tol=1e-45,
                   normalize=None, random_state=1, init_method="random")
m.fit(Xt, C0.clone())
C = m.cluster_centers_.float().contiguous()

Xd, Cd = Xt.double(), C.double()
Dex = (Xd*Xd).sum(1, keepdim=True) - 2*(Xd @ Cd.T) + (Cd*Cd).sum(1)   # FP64 truth
Dex = torch.cdist(Xd, Cd).pow(2)                                       # less cancellation
arg_ex = Dex.argmin(1)
scale = Dex.abs().clamp_min(1e-300)

print(f"n={n} d={d} k={k}, converged centroids, FP64 ground truth")
print(f"  {'kappa':>9} {'max rel err':>13} {'mean rel err':>13} "
      f"{'argmin wrong':>13} {'%':>8} {'ms':>8}")
import time
def bench(f, reps=15):
    f(); torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(reps): o=f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)*1e3/reps, o

for kap in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 1e6):
    ms, Dk = bench(lambda: E.pairwise_euclidean_fp16_fp32(Xt, C, kap))
    err = (Dk.double() - Dex).abs() / scale
    wrong = int((Dk.argmin(1) != arg_ex).sum().item())
    print(f"  {kap:9.4g} {err.max().item():13.3e} {err.mean().item():13.3e} "
          f"{wrong:13d} {100.0*wrong/n:7.4f}% {ms:8.3f}")
ms32, D32 = bench(lambda: E.pairwise_euclidean_single(Xt, C))
err = (D32.double() - Dex).abs() / scale
w = int((D32.argmin(1) != arg_ex).sum().item())
print(f"  {'fp32':>9} {err.max().item():13.3e} {err.mean().item():13.3e} "
      f"{w:13d} {100.0*w/n:7.4f}% {ms32:8.3f}")
