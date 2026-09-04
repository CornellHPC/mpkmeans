#!/usr/bin/env python3
"""Turn the evaluation CSVs into plots.  Driven by 3_plot.sh; see that for the
venv it needs.

Everything here is normalised PER ITERATION.  The schemes stop on their own
convergence criteria and so run different numbers of iterations on the same
problem, which makes run totals incomparable -- the CSV carries both `iters`
and `iters_fp32` so the division can be done honestly.

Two time bases, deliberately kept apart:
  ms_dist   the distance step only (prep + GEMM + argmin/exclusion), which is
            what the exclusion conditions actually change.
  ms_total  the whole fit including the centroid update.  This is the only
            column the cuvs row can be compared on, because cuVS is one opaque
            library call, so cuvs appears only in the end-to-end plot.
"""
import argparse, csv, math, os, sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# the mixed-precision schemes, in the order they should appear in a legend
CONDS = ["(3)", "(6)", "(3)+(6)", "(3)->(6)", "raw", "mw", "rt-base", "cuvs",
         "rt-mp"]
COLORS = {
    "(3)": "#4c72b0", "(6)": "#dd8452", "(3)+(6)": "#55a868",
    "(3)->(6)": "#c44e52", "raw": "#8172b3", "mw": "#937860",
    "rt-base": "#da8bc3", "cuvs": "#8c8c8c", "rt-mp": "#000000",
}

# rt-mp rows come from eval/rt_baseline_mp.py -- the arXiv:2407.12208 authors'
# own package.  Its inertia_fp32/ms_total_fp32/iters_fp32 columns are that
# package's OWN fp32 fit, not mpkMeansFP32, so a "speedup vs fp32" computed
# from them measures something different from every other row and must not
# share an axis with them.  It appears in the absolute-time plots instead,
# where no reference is implied.
WITHIN_PKG_REF = {"rt-mp"}
RATIO_CONDS = [c for c in CONDS if c not in WITHIN_PKG_REF]
NUM = {"n", "d", "k", "std", "box", "seed", "zscore", "iters", "eps",
       "hp_baseline", "hp_reference", "hp_update", "hp_total",
       "pct_eliminated", "pct_reference", "pct_update", "pct_cond3",
       "pct_cond6", "pct_cond3_only", "pct_cond6_only", "violations",
       "label_diff", "inertia", "inertia_fp32", "rel_inertia", "ms_dist",
       "ms_dist_fp32", "speedup", "ms_prep", "ms_gemm", "ms_argmin",
       "ms_hpupdate", "ms_assign", "ms_total", "ms_total_fp32", "iters_fp32"}


def load(results_dir):
    rows = []
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".csv"))
    if not files:
        sys.exit(f"no CSVs in {results_dir} -- has run_all.sh finished?")
    for fn in files:
        path = os.path.join(results_dir, fn)
        with open(path) as f:
            for r in csv.DictReader(f):
                if not r.get("cond"):
                    continue
                for key in list(r):
                    if key in NUM and r[key] not in (None, ""):
                        try:
                            r[key] = float(r[key])
                        except ValueError:
                            pass
                r["_src"] = fn
                rows.append(r)
    print(f"loaded {len(rows)} rows from {len(files)} file(s)")
    return rows


def per_iter(r, base):
    """ms per iteration for this row, and for the fp32 reference beside it."""
    it = max(r.get("iters", 0) or 0, 1)
    itf = max(r.get("iters_fp32", 0) or 0, 1)
    return r[base] / it, r[f"{base}_fp32"] / itf


def speedup(r, base="ms_dist"):
    mine, ref = per_iter(r, base)
    return ref / mine if mine > 0 else float("nan")


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.basename(path))
    return path


def line_panels(rows, out, xkey, fname, title, ykey, ylabel, logy=False,
                hline=None, conds=None, base="ms_dist"):
    """One panel per (box, n); a line per scheme; x is d or k."""
    conds = conds or [c for c in RATIO_CONDS if c != "cuvs"]
    groups = sorted({(r["box"], r["n"]) for r in rows})
    if not groups:
        return None
    ncol = min(4, len(groups))
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False, sharey=True)
    for ax, (box, n) in zip(axes.flat, groups):
        for cond in conds:
            pts = defaultdict(list)
            for r in rows:
                if r["box"] == box and r["n"] == n and r["cond"] == cond:
                    v = ykey(r) if callable(ykey) else r[ykey]
                    if v == v:                      # drop NaN
                        pts[r[xkey]].append(v)
            if not pts:
                continue
            xs = sorted(pts)
            ys = [float(np.median(pts[x])) for x in xs]
            ax.plot(xs, ys, "o-", ms=4, lw=1.5, label=cond,
                    color=COLORS.get(cond))
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"separation b={box:g}, n={int(n):,}", fontsize=9)
        ax.set_xlabel(xkey)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
    fig.suptitle(title, y=1.07, fontsize=11)
    return save(fig, out, fname)


def bars_by_dataset(rows, out, fname, title, valfn, ylabel, hline=None,
                    conds=None):
    conds = conds or RATIO_CONDS
    dsets = sorted({r["dataset"] for r in rows})
    if not dsets:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(dsets) + 3), 4))
    width = 0.8 / len(conds)
    xs = np.arange(len(dsets))
    for i, cond in enumerate(conds):
        vals = []
        for ds in dsets:
            v = [valfn(r) for r in rows
                 if r["dataset"] == ds and r["cond"] == cond]
            v = [x for x in v if x == x]
            vals.append(float(np.median(v)) if v else np.nan)
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width, label=cond,
               color=COLORS.get(cond))
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(dsets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    return save(fig, out, fname)


def paired_effect(rows, out, fname, title, key, on_label, off_label):
    """Median per-iteration speedup with the flag on vs off, per scheme."""
    conds = [c for c in RATIO_CONDS if c != "cuvs"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    xs = np.arange(len(conds))
    for j, (state, lab) in enumerate([(0, off_label), (1, on_label)]):
        vals = []
        for cond in conds:
            v = [speedup(r) for r in rows
                 if r["cond"] == cond and _flag(r, key) == state]
            v = [x for x in v if x == x]
            vals.append(float(np.median(v)) if v else np.nan)
        ax.bar(xs + j * width - width / 2, vals, width, label=lab)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(conds, rotation=15)
    ax.set_ylabel("distance-step speedup vs fp32, per iteration")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(frameon=False)
    return save(fig, out, fname)


def _flag(r, key):
    return 1 if (r["accum"] == "fp16" if key == "accum" else r["zscore"] == 1) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="directory of CSVs")
    ap.add_argument("--out", required=True, help="directory for the plots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load(args.results)
    synth = [r for r in rows if r["dataset"] == "blobs"]
    real = [r for r in rows if r["dataset"] != "blobs"]

    # a run that violated a bound is not a result, it is a bug -- say so loudly
    bad = [r for r in rows if (r.get("violations") or 0) > 0]
    if bad:
        print(f"\n  !! {len(bad)} row(s) report bound violations -- "
              f"these are correctness failures, not slow runs:")
        for r in bad[:10]:
            print(f"     {r['_src']} {r['cond']} n={r['n']:.0f} d={r['d']:.0f} "
                  f"k={r['k']:.0f} violations={r['violations']:.0f}")

    print("\nsynthetic:")
    if synth:
        # the d sweep and the k sweep are separate experiments: within one, the
        # other axis is fixed, so plotting against the wrong one would average
        # over a variable rather than hold it still
        for axis, other in (("d", "k"), ("k", "d")):
            # hold the other axis at its most common value -- that is the pivot
            # 2_gen_jobs.sh swept around, and mixing pivots would average over
            # the very variable the panel is supposed to hold still
            fixed = sorted({r[other] for r in synth})
            pivot = max(fixed, key=lambda v: sum(1 for rr in synth if rr[other] == v))
            sel = [r for r in synth if r[other] == pivot]
            line_panels(sel, args.out, axis, f"speedup_vs_{axis}.png",
                        f"distance-step speedup vs fp32, per iteration "
                        f"(sweep over {axis})",
                        speedup, "speedup", hline=1.0)
            # mw is excluded on purpose: it refines the whole distance matrix
            # one fp16 cross product at a time rather than flagging entries, so
            # "entries eliminated" is not a quantity it has, and the column
            # reads ~0 (or slightly negative) for reasons that say nothing
            # about how well it is doing.  Its cost is products per iteration.
            line_panels(sel, args.out, axis, f"eliminated_vs_{axis}.png",
                        f"high-precision distance evaluations eliminated "
                        f"(sweep over {axis}; mw excluded, see note)",
                        "pct_eliminated", "% eliminated",
                        conds=[c for c in CONDS if c not in ("cuvs", "mw")])
        line_panels(synth, args.out, "k", "endtoend_vs_k_synth.png",
                    "end-to-end speedup vs fp32, per iteration (whole fit)",
                    lambda r: speedup(r, "ms_total"), "speedup", hline=1.0,
                    conds=CONDS)
    else:
        print("  (no synthetic rows)")

    print("\nreal datasets:")
    if real:
        bars_by_dataset(real, args.out, "real_speedup.png",
                        "distance-step speedup vs fp32, per iteration "
                        "(median over k)",
                        speedup, "speedup", hline=1.0,
                        conds=[c for c in CONDS if c != "cuvs"])
        bars_by_dataset(real, args.out, "real_endtoend.png",
                        "end-to-end speedup vs fp32, per iteration "
                        "(whole fit, median over k)",
                        lambda r: speedup(r, "ms_total"), "speedup", hline=1.0)
        bars_by_dataset(real, args.out, "real_eliminated.png",
                        "high-precision distance evaluations eliminated "
                        "(median over k)",
                        lambda r: r["pct_eliminated"], "% eliminated",
                        conds=[c for c in CONDS if c not in ("cuvs", "raw", "mw")])
        bars_by_dataset(real, args.out, "real_accuracy.png",
                        "inertia relative to the fp32 reference (median over k)",
                        lambda r: r["rel_inertia"], "|dSSE| / SSE_fp32")
    else:
        print("  (no real-dataset rows)")

    # Absolute wall time per iteration is the one basis on which our schemes,
    # cuVS and the authors' package can all be put side by side: it implies no
    # reference and no shared convergence rule.
    def ms_per_iter(r):
        it = max(r.get("iters", 0) or 0, 1)
        return r["ms_total"] / it

    if real:
        bars_by_dataset(real, args.out, "real_ms_per_iter.png",
                        "wall time per iteration, whole fit "
                        "(absolute; median over k)",
                        ms_per_iter, "ms / iteration", conds=CONDS)
    if synth:
        line_panels(synth, args.out, "k", "ms_per_iter_vs_k_synth.png",
                    "wall time per iteration, whole fit (absolute)",
                    ms_per_iter, "ms / iteration", logy=True, conds=CONDS)

    print("\nflag effects:")
    paired_effect(rows, args.out, "accum_effect.png",
                  "effect of the low-precision accumulator", "accum",
                  "fp16 accumulate", "fp32 accumulate")
    paired_effect(rows, args.out, "zscore_effect.png",
                  "effect of z-score normalization", "zscore",
                  "--zscore", "raw features")

    # ------------------------------------------------------------ summary --
    summary = os.path.join(args.out, "summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "box", "n", "accum", "zscore", "cond",
                    "runs", "median_speedup_dist", "median_speedup_total",
                    "median_pct_eliminated", "median_rel_inertia",
                    "max_violations"])
        keyf = lambda r: (r["dataset"], r["box"], r["n"], r["accum"],
                          r["zscore"], r["cond"])
        groups = defaultdict(list)
        for r in rows:
            groups[keyf(r)].append(r)
        for kk in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
            g = groups[kk]
            med = lambda vs: float(np.median(vs)) if vs else float("nan")
            w.writerow(list(kk) + [
                len(g),
                f"{med([speedup(r) for r in g if speedup(r) == speedup(r)]):.4f}",
                f"{med([speedup(r,'ms_total') for r in g]):.4f}",
                f"{med([r['pct_eliminated'] for r in g]):.4f}",
                f"{med([r['rel_inertia'] for r in g]):.3e}",
                int(max((r.get('violations') or 0) for r in g)),
            ])
    print("\n  wrote", os.path.basename(summary))
    print(f"\nplots in {args.out}")


if __name__ == "__main__":
    main()
