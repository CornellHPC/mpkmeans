#!/usr/bin/env python3
"""Plots for the rectangular-shape iteration experiment (gaussian data).

All runtimes come from the 100-iteration steady-state loop: A split once, only
the small operand re-split each iteration.

Two figure families per shape:

  split_<shape>.png   bf16-split and multiword, each in the original
                      per-product ("naive") form and the concatenated-k form.
                      Error, speedup over FP64 and FP32, and a runtime
                      breakdown at the full product count.
  ozaki_<shape>.png   Ozaki I on its own -- its product range (1..36) and its
                      accuracy are two orders apart from the others, so it does
                      not share an axis usefully.

  module load python/3.13-26.8.0
  python3 scripts/plot_rect.py results/rect-gaussian.csv
"""
import csv, sys, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPLIT = [("bf16-split",        "bf16-split, per-product", "#4C72B0", "--", "o"),
         ("bf16-split-concat", "bf16-split, concat",      "#2A4C7D", "-",  "o"),
         ("multiword",         "multiword, per-product",  "#DD8452", "--", "s"),
         ("multiword-concat",  "multiword, concat",       "#A9542F", "-",  "s")]
PARTS = [("ms_scale","scaling","#C44E52"),
         ("ms_split","splitting","#8172B3"),
         ("ms_gemm","GEMMs","#4C72B0")]
REF = [("fp64","cublasDgemm","k","--"), ("fp32","cublasSgemm","dimgray",":"),
       ("bf16-gemm","plain BF16 GEMM","#4C72B0","-."),
       ("fp16-gemm","plain FP16 GEMM","#DD8452","-.")]

def main(paths, outdir="results/plots"):
    os.makedirs(outdir, exist_ok=True)
    rows=[]
    for p in paths:
        with open(p) as f: rows += list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in rows: groups[(int(r["m"]),int(r["n"]),int(r["k"]))].append(r)

    made=[]
    for (m,n,k),g in sorted(groups.items(), key=lambda x:(x[0][1],x[0][2])):
        base={r["method"]:r for r in g if r["method"] in [x[0] for x in REF]}
        if "fp64" not in base or "fp32" not in base: continue
        t64=float(base["fp64"]["ms_per_iter"]); t32=float(base["fp32"]["ms_per_iter"])
        shape=f"{m}x{n}x{k}"
        title=f"gaussian   C = A·B   {m}×{n}×{k}   100 iterations, A split once   A100-SXM4-40GB"

        def series(meth):
            return sorted([r for r in g if r["method"]==meth], key=lambda r:int(r["products"]))

        def refs(ax, field, logy):
            for key,lbl,col,ls in REF:
                if key not in base: continue
                v=float(base[key][field])
                ax.axhline(v, ls=ls, c=col, lw=1.3, alpha=.85,
                           label=f"{lbl}  {v:.2e}" if logy else lbl)

        # ---------------- split schemes ----------------
        fig,ax=plt.subplots(2,2,figsize=(15,10))
        fig.suptitle(title,fontsize=12)

        a=ax[0][0]
        for key,lbl,col,ls,mk in SPLIT:
            s=series(key)
            if not s: continue
            a.plot([int(r["products"]) for r in s],
                   [max(float(r["rel_err"]),1e-20) for r in s],
                   ls=ls,marker=mk,color=col,label=lbl,lw=1.8,ms=5)
        refs(a,"rel_err",True)
        a.set_yscale("log"); a.set_xlabel("products accumulated")
        a.set_ylabel("relative error vs double-double reference")
        a.set_title("accuracy"); a.grid(alpha=.3,which="both"); a.legend(fontsize=7)

        for j,(tb,nm,fld) in enumerate([(t64,"FP64 (cublasDgemm)","vs_fp64"),
                                        (t32,"FP32 (cublasSgemm)","vs_fp32")]):
            a=ax[0][1] if j==0 else ax[1][0]
            present=[x for x in SPLIT if series(x[0])]
            allx=sorted({int(r["products"]) for x in present for r in series(x[0])})
            w=0.8/max(len(present),1)
            for i,(key,lbl,col,ls,mk) in enumerate(present):
                s=series(key)
                xs=np.array([allx.index(int(r["products"])) for r in s],dtype=float)
                a.bar(xs+(i-(len(present)-1)/2)*w,[float(r[fld]) for r in s],w,color=col,label=lbl)
            a.axhline(1.0,c="k",lw=1.2)
            a.set_xticks(range(len(allx))); a.set_xticklabels(allx)
            a.set_xlabel("products accumulated"); a.set_ylabel(f"speedup over {nm.split()[0]}")
            a.set_title(f"speedup over {nm}",fontsize=10)
            a.grid(alpha=.3,axis="y"); a.legend(fontsize=7)

        a=ax[1][1]
        present=[x for x in SPLIT if series(x[0])]
        labels=[]; 
        for i,(key,lbl,col,ls,mk) in enumerate(present):
            s=series(key)
            if not s: continue
            r=s[-1]                                   # full product count
            bottom=0.0
            for fld,plbl,pcol in PARTS:
                v=float(r[fld])
                if v<=0: continue
                a.bar(i,v,0.62,bottom=bottom,color=pcol,
                      label=plbl if i==0 else None)
                bottom+=v
            a.plot([i],[float(r["ms_per_iter"])],"kD",ms=6,
                   label="measured loop" if i==0 else None)
            labels.append(lbl.replace(", ","\n"))
        refs(a,"ms_per_iter",False)
        a.set_xticks(range(len(labels))); a.set_xticklabels(labels,fontsize=7)
        a.set_ylabel("ms / iteration")
        a.set_title("runtime breakdown at full products\n(stages timed separately; "
                    "diamond = fused loop)",fontsize=9)
        a.grid(alpha=.3,axis="y"); a.legend(fontsize=7)

        fig.tight_layout(rect=[0,0,1,0.95])
        o=os.path.join(outdir,f"split_{shape}.png"); fig.savefig(o,dpi=120); plt.close(fig); made.append(o)

        # ---------------- Ozaki I, on its own ----------------
        s=series("ozaki1")
        if s:
            fig,ax=plt.subplots(1,3,figsize=(18,5))
            fig.suptitle("Ozaki I (INT8, 8 slices) — "+title,fontsize=12)
            xs=[int(r["products"]) for r in s]

            a=ax[0]
            a.plot(xs,[max(float(r["rel_err"]),1e-20) for r in s],"o-",color="#55A868",lw=1.9,ms=5,label="Ozaki I")
            refs(a,"rel_err",True)
            a.set_yscale("log"); a.set_xlabel("products accumulated")
            a.set_ylabel("relative error"); a.set_title("accuracy")
            a.grid(alpha=.3,which="both"); a.legend(fontsize=7)

            for j,(nm,fld) in enumerate([("FP64 (cublasDgemm)","vs_fp64"),("FP32 (cublasSgemm)","vs_fp32")]):
                a=ax[1+j]
                a.bar(range(len(xs)),[float(r[fld]) for r in s],0.7,color="#55A868")
                a.axhline(1.0,c="k",lw=1.2)
                a.set_xticks(range(len(xs))); a.set_xticklabels(xs,fontsize=8)
                a.set_xlabel("products accumulated"); a.set_ylabel(f"speedup over {nm.split()[0]}")
                a.set_title(f"speedup over {nm}",fontsize=10); a.grid(alpha=.3,axis="y")
            fig.tight_layout(rect=[0,0,1,0.93])
            o=os.path.join(outdir,f"ozaki_{shape}.png"); fig.savefig(o,dpi=120); plt.close(fig); made.append(o)

    for f in made: print("wrote",f)
    print(f"\n{len(made)} figures")

if __name__=="__main__":
    if len(sys.argv)<2: print(__doc__); sys.exit(1)
    main(sys.argv[1:])
