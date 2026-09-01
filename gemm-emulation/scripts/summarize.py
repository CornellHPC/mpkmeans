#!/usr/bin/env python3
"""Format a mpemu_bench CSV into the markdown tables used in RESULTS.md."""
import csv, sys
from collections import defaultdict

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def fnum(r, k):
    try:    return float(r[k])
    except (ValueError, KeyError): return float("nan")

def perf_table(rows, tag):
    sel = [r for r in rows if r["shape"] == tag]
    shapes, base = [], {}
    for r in sel:
        key = (int(r["m"]), int(r["n"]), int(r["k"]))
        if key not in shapes: shapes.append(key)
        if r["method"] == "cublasSgemm": base[key] = fnum(r, "tflops_gemm")
    macs = sorted({int(r["macs"]) for r in sel if r["method"] == "mpemu"})

    out = ["| shape (m×n×k) | SGEMM | " +
           " | ".join(f"{m} MAC" for m in macs) + " |",
           "|---|---|" + "---|" * len(macs)]
    for key in shapes:
        cells = [f"`{key[0]}×{key[1]}×{key[2]}`", f"{base.get(key, float('nan')):.1f}"]
        for m in macs:
            hit = [r for r in sel if (int(r["m"]), int(r["n"]), int(r["k"])) == key
                   and r["method"] == "mpemu" and int(r["macs"]) == m]
            if hit:
                t, s = fnum(hit[0], "tflops_gemm"), fnum(hit[0], "speedup_gemm")
                cells.append(f"{t:.1f} (**{s:.2f}×**)")
            else:
                cells.append("—")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)

def err_table(rows):
    out = ["| shape (m×n×k) | SGEMM | 1 MAC | 3 MAC | 6 MAC | 9 MAC |",
           "|---|---|---|---|---|---|"]
    seen = []
    for r in rows:
        key = (int(r["m"]), int(r["n"]), int(r["k"]))
        if key not in seen and fnum(r, "rel_frob") > 0: seen.append(key)
    for key in seen:
        grp = [r for r in rows if (int(r["m"]), int(r["n"]), int(r["k"])) == key]
        sg = [r for r in grp if r["method"] == "cublasSgemm"]
        if not sg or fnum(sg[0], "rel_frob") <= 0: continue
        cells = [f"`{key[0]}×{key[1]}×{key[2]}`", f"{fnum(sg[0],'rel_frob'):.2e}"]
        for m in (1, 3, 6, 9):
            hit = [r for r in grp if r["method"] == "mpemu" and int(r["macs"]) == m]
            cells.append(f"{fnum(hit[0],'rel_frob'):.2e}" if hit else "—")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)

if __name__ == "__main__":
    rows = load(sys.argv[1])
    for tag in ["square", "k-thin", "n-thin", "m-thin", "k-fat"]:
        if any(r["shape"] == tag for r in rows):
            print(f"\n### {tag}  (effective FP32 TFLOP/s, GEMM only)\n")
            print(perf_table(rows, tag))
    print("\n### Forward error (relative Frobenius vs FP64)\n")
    print(err_table(rows))
