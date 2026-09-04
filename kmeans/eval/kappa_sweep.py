"""Accuracy and cost of mp-kmeans' fp16_fp32 kernel against kappa.

Same data, same initial centroids, same fixed iteration count for every kappa,
so the only thing varying is the reliability test's safety factor.  Inertia is
recomputed in FP64 from the final centroids and labels -- the same quantity the
benchmark reports -- and scored against this package's own fp32 kernel run
through the identical loop.
"""
import sys, time, numpy as np, torch
from mp_kmeans.clustering import KMeansPlusPlus
from mp_kmeans import euclidean_cuda as E

path, d, k, n_sub, ITERS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), 20
X = np.fromfile(path, dtype=np.float32).reshape(-1, d)
rng = np.random.default_rng(1)
if 0 < n_sub < X.shape[0]:
    X = np.ascontiguousarray(X[np.sort(rng.choice(X.shape[0], n_sub, replace=False))])
n = X.shape[0]
Xt = torch.from_numpy(X).cuda().contiguous()
C0 = torch.from_numpy(np.ascontiguousarray(X[rng.choice(n, k, replace=False)])).cuda().contiguous()

def fit(kernel, kappa):
    m = KMeansPlusPlus(n_clusters=k, kernel=kernel, kappa=kappa, max_iter=ITERS,
                       tol=1e-45, normalize=None, random_state=1, init_method="random")
    torch.cuda.synchronize(); t = time.perf_counter()
    m.fit(Xt, C0.clone()); torch.cuda.synchronize()
    return m, (time.perf_counter() - t) * 1e3

def inertia(m):
    return float((((Xt.double() - m.cluster_centers_.double()[m.labels_]) ** 2).sum()).item())

def dist_ms(kappa, reps=15):
    C = C0.clone()
    f = lambda: E.pairwise_euclidean_fp16_fp32(Xt, C, kappa)
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / reps

fit("fp32", 5.0)                                  # full-size warm-up
ref, ref_ms = fit("fp32", 5.0)
ref_I = inertia(ref)
print(f"n={n} d={d} k={k}, {ITERS} fixed iterations, shared centroids")
print(f"reference: their fp32 kernel, inertia={ref_I:.10e}, {ref_ms:.1f} ms\n")
print(f"  {'kappa':>9} {'rel inertia':>13} {'signed':>13} {'labels diff':>12} "
      f"{'%':>7} {'dist ms/it':>11} {'fit ms':>9}")
for kap in (0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 1e3, 1e6):
    m, ms = fit("fp16_fp32", kap)
    I = inertia(m)
    rel = (I - ref_I) / abs(ref_I)
    nd = int((m.labels_ != ref.labels_).sum().item())
    print(f"  {kap:9.4g} {abs(rel):13.3e} {rel:+13.3e} {nd:12d} "
          f"{100.0*nd/n:6.3f}% {dist_ms(kap):11.3f} {ms:9.1f}")
