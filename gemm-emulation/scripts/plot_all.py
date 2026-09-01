#!/usr/bin/env python3
"""Plots for the three-scheme refinement comparison.

Each scheme is split once, as many times as it needs for full accuracy in its
target format; the x axis is then the number of PRODUCTS accumulated into the
output. Per (distribution, shape) one figure with six panels:

  top    error vs products (log), one line per scheme, with the cublasDgemm
         and cublasSgemm error levels as horizontal references
         speedup over FP64 and over FP32, as bars vs products
  bottom runtime breakdown per scheme: the one-off costs (scaling, splitting)
         stacked under the cumulative per-product costs (GEMM, and the
         INT32->FP64 fold for Ozaki I), with the cuBLAS baselines as lines

  module load python/3.13-26.8.0
  python3 scripts/plot_all.py results/all-*.csv
"""
import csv, sys, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCHEMES = [("bf16",   "bf16 (Henry) 3 splits -> FP32",        "#4C72B0"),
           ("fp16mw", "multiword fp16 (Fasi) 2 splits -> FP32","#DD8452"),
           ("ozaki1", "Ozaki I INT8 (Ootomo) 8 slices -> FP64","#55A868")]
PARTS = [("ms_scale", "scaling / exponents", "#C44E52"),
         ("ms_split", "splitting",           "#8172B3"),
         ("ms_gemm_cum", "GEMMs (cumulative)", "#4C72B0"),
         ("ms_fold_cum", "INT32->FP64 fold",   "#937860")]

def main(paths, outdir="results/plots"):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for p in paths:
        with open(p) as f: rows += list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["dist"], int(r["m"]), int(r["n"]), int(r["k"]))].append(r)

    made = []
    for (dist, m, n, k), g in sorted(groups.items()):
        base = {r["scheme"]: r for r in g if r["scheme"] in ("fp64","fp32")}
        if len(base) < 2: continue
        t64, e64 = float(base["fp64"]["ms_total"]), float(base["fp64"]["rel_err"])
        t32, e32 = float(base["fp32"]["ms_total"]), float(base["fp32"]["rel_err"])

        data = {}
        for key,_,_ in SCHEMES:
            pts = sorted([r for r in g if r["scheme"]==key],
                         key=lambda r: int(r["products"]))
            if pts: data[key] = pts

        fig, ax = plt.subplots(2, 3, figsize=(21, 10))
        fig.suptitle(f"{dist} data     C = A·B     {m}×{n}×{k}     A100-SXM4-40GB",
                     fontsize=14)

        # ---------------- error ----------------
        a = ax[0][0]
        for key,label,col in SCHEMES:
            if key not in data: continue
            xs=[int(r["products"]) for r in data[key]]
            ys=[max(float(r["rel_err"]),1e-20) for r in data[key]]
            a.plot(xs,ys,"o-",color=col,label=label,lw=1.8,ms=4)
        a.axhline(e64, ls="--", c="k", lw=1.3, label=f"cublasDgemm  {e64:.1e}")
        a.axhline(e32, ls=":", c="dimgray", lw=1.7, label=f"cublasSgemm  {e32:.1e}")
        a.set_yscale("log"); a.set_xlabel("products accumulated")
        a.set_ylabel("relative error vs double-double reference")
        a.set_title("accuracy as products are added")
        a.grid(alpha=.3, which="both"); a.legend(fontsize=7.5)

        # ---------------- speedups ----------------
        for j,(tb,name) in enumerate([(t64,"FP64 (cublasDgemm)"),(t32,"FP32 (cublasSgemm)")]):
            a = ax[0][1+j]
            present=[s for s in SCHEMES if s[0] in data]
            w = 0.8/max(len(present),1)
            for i,(key,label,col) in enumerate(present):
                xs=np.array([int(r["products"]) for r in data[key]],dtype=float)
                sp=np.array([tb/float(r["ms_total"]) for r in data[key]])
                a.bar(xs+(i-(len(present)-1)/2)*w, sp, w, color=col, label=label)
            a.axhline(1.0,c="k",lw=1.2)
            a.set_xlabel("products accumulated")
            a.set_ylabel(f"speedup over {name.split()[0]}")
            a.set_title(f"speedup over {name}\n(total: one-off costs + products)",fontsize=10)
            a.grid(alpha=.3,axis="y"); a.legend(fontsize=7.5)

        # ---------------- breakdowns ----------------
        for i,(key,label,col) in enumerate(SCHEMES):
            a = ax[1][i]
            if key not in data:
                a.set_visible(False); continue
            pts=data[key]
            xs=np.array([int(r["products"]) for r in pts],dtype=float)
            bottom=np.zeros(len(pts))
            for field,plabel,pcol in PARTS:
                vals=np.array([float(r[field]) for r in pts])
                if vals.max()<=0: continue
                a.bar(xs,vals,0.82,bottom=bottom,color=pcol,label=plabel)
                bottom+=vals
            a.axhline(t64,ls="--",c="k",lw=1.2,label="cublasDgemm")
            a.axhline(t32,ls=":",c="dimgray",lw=1.6,label="cublasSgemm")
            a.set_xlabel("products accumulated"); a.set_ylabel("time (ms)")
            a.set_title(f"runtime breakdown — {label}",fontsize=9.5)
            a.grid(alpha=.3,axis="y"); a.legend(fontsize=7)

        fig.tight_layout(rect=[0,0,1,0.95])
        out=os.path.join(outdir,f"{dist}_{m}x{n}x{k}.png")
        fig.savefig(out,dpi=120); plt.close(fig); made.append(out)

    for f in made: print("wrote",f)
    print(f"\n{len(made)} figures, 6 panels each")

if __name__=="__main__":
    if len(sys.argv)<2: print(__doc__); sys.exit(1)
    main(sys.argv[1:])
