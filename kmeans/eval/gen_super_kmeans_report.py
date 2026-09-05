#!/usr/bin/env python3
"""Turn eval/runs/super_kmeans_sweep/results.csv into a self-contained HTML
report: one row per (dataset, k, kappa) from rt_baseline_mp.py, run over
every SuperKMeans vector-indexing dataset scripts/fetch_superkmeans_datasets.sh
can produce (see eval/run_super_kmeans_sweep.sh -- NOT the three-dataset
subset the eval/ pipeline itself uses).

Usage: gen_super_kmeans_report.py [results.csv] [-o report.html]
"""
import argparse, csv, json, math, os, sys
from collections import defaultdict

K_ORDER = [32, 64, 128, 256, 512, 1024]
KAPPAS = [1, 5]


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        cond = r["cond"]
        if not cond.startswith("rt-mp-k"):
            continue
        kap = float(cond[len("rt-mp-k"):])
        try:
            ms = float(r["ms_total"]); ms32 = float(r["ms_total_fp32"])
            iters = int(r["iters"]); iters32 = int(r["iters_fp32"])
            rel = float(r["rel_inertia"])
            n = int(r["n"]); k = int(r["k"]); d = int(r["d"])
            label_diff = int(r["label_diff"])
        except (ValueError, KeyError):
            continue
        dset_name = r["dataset"]
        if dset_name.startswith("data_"):
            dset_name = dset_name[len("data_"):]
        if dset_name.endswith(".bin"):
            dset_name = dset_name[:-len(".bin")]
        out.append(dict(
            dataset=dset_name,
            n=n, d=d, k=k, kappa=kap, iters=iters, iters_fp32=iters32,
            ms_total=ms, ms_total_fp32=ms32,
            ratio=(ms / ms32 if ms32 > 0 else float("nan")),
            rel_inertia=rel, label_diff=label_diff,
            label_diff_pct=100.0 * label_diff / n if n else float("nan"),
        ))
    return out


def fmt_ms(ms):
    return f"{ms/1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"


def build(rows, out_path):
    by_dataset = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)
    datasets = sorted(by_dataset, key=lambda d: by_dataset[d][0]["d"])

    n_cells = len(rows)
    slower = [r for r in rows if r["ratio"] > 1.0]
    faster = [r for r in rows if r["ratio"] <= 1.0]
    worst = max(rows, key=lambda r: r["ratio"]) if rows else None
    best = min(rows, key=lambda r: r["ratio"]) if rows else None
    med_ratio = sorted(r["ratio"] for r in rows)[len(rows)//2] if rows else float("nan")

    chart_data = {
        d: {
            str(kap): [
                {"k": r["k"], "ratio": r["ratio"], "rel_inertia": r["rel_inertia"],
                 "ms": r["ms_total"], "ms32": r["ms_total_fp32"]}
                for r in sorted(by_dataset[d], key=lambda r: r["k"])
                if r["kappa"] == kap
            ] for kap in KAPPAS
        } for d in datasets
    }
    meta = {d: dict(n=by_dataset[d][0]["n"], dd=by_dataset[d][0]["d"]) for d in datasets}

    table_rows = []
    for d in datasets:
        for k in K_ORDER:
            for kap in KAPPAS:
                match = [r for r in by_dataset[d] if r["k"] == k and r["kappa"] == kap]
                if not match:
                    continue
                r = match[0]
                table_rows.append(r)

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;")

    rows_html = []
    prev_d = None
    for r in table_rows:
        first = r["dataset"] != prev_d
        prev_d = r["dataset"]
        rows_html.append(f"""<tr class="{'new-ds' if first else ''}">
<td class="c-ds">{esc(r['dataset']) if first else ''}</td>
<td class="num">{r['k']}</td>
<td class="num">{r['kappa']:g}</td>
<td class="num">{r['iters']}</td>
<td class="num">{fmt_ms(r['ms_total'])}</td>
<td class="num">{fmt_ms(r['ms_total_fp32'])}</td>
<td class="num {'bad' if r['ratio']>1 else 'good'}">{r['ratio']:.2f}&times;</td>
<td class="num">{r['rel_inertia']:.2e}</td>
<td class="num">{r['label_diff_pct']:.3f}%</td>
</tr>""")

    dataset_cards = []
    for d in datasets:
        m = meta[d]
        dataset_cards.append(f"""
<div class="ds-card">
  <div class="ds-head">
    <span class="ds-name">{esc(d)}</span>
    <span class="ds-dims">n={m['n']:,}&ensp;d={m['dd']}</span>
  </div>
  <div class="ds-charts">
    <div class="chart" data-dataset="{esc(d)}" data-metric="ratio"></div>
    <div class="chart" data-dataset="{esc(d)}" data-metric="rel_inertia"></div>
  </div>
</div>""")

    html = f"""<title>SuperKMeans mp-kmeans Sweep</title>
<style>
:root {{
  --bg: #f6f7f9;
  --surface: #ffffff;
  --surface-2: #eef0f3;
  --ink: #1b2430;
  --muted: #5b6472;
  --line: #dde1e7;
  --mixed: #c9762f;      /* fp16_fp32 kernel */
  --mixed-soft: #f2ddc4;
  --ref: #3d6fa8;        /* fp32 reference */
  --ref-soft: #d8e3ef;
  --bad: #b3452f;
  --good: #3d7a52;
  --font-display: "Newsreader", Georgia, "Times New Roman", serif;
  --font-body: "Source Sans 3", -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #12161d;
    --surface: #1a1f28;
    --surface-2: #212733;
    --ink: #e8eaee;
    --muted: #9aa4b2;
    --line: #2c3340;
    --mixed: #e2934f;
    --mixed-soft: #3a2c1a;
    --ref: #7ba7d9;
    --ref-soft: #1e2c3b;
    --bad: #e07258;
    --good: #6fbf8b;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #12161d;
  --surface: #1a1f28;
  --surface-2: #212733;
  --ink: #e8eaee;
  --muted: #9aa4b2;
  --line: #2c3340;
  --mixed: #e2934f;
  --mixed-soft: #3a2c1a;
  --ref: #7ba7d9;
  --ref-soft: #1e2c3b;
  --bad: #e07258;
  --good: #6fbf8b;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font-body);
  margin: 0;
  padding: 0 1.5rem 4rem;
}}
.wrap {{ max-width: 980px; margin: 0 auto; }}
header {{
  padding: 3rem 0 1.75rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 2rem;
}}
h1 {{
  font-family: var(--font-display);
  font-size: 2.1rem;
  font-weight: 600;
  margin: 0 0 .4rem;
  text-wrap: balance;
  letter-spacing: -0.01em;
}}
.subtitle {{
  color: var(--muted);
  font-size: 1.02rem;
  max-width: 60ch;
  line-height: 1.5;
}}
.meta-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  margin-top: 1.25rem;
  font-family: var(--font-mono);
  font-size: .8rem;
  color: var(--muted);
}}
.meta-strip b {{ color: var(--ink); font-weight: 600; }}
.legend {{
  display: flex;
  gap: 1.5rem;
  font-size: .85rem;
  margin-top: .9rem;
}}
.legend span {{ display: inline-flex; align-items: center; gap: .4rem; }}
.swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}

h2 {{
  font-family: var(--font-display);
  font-size: 1.3rem;
  font-weight: 600;
  margin: 0 0 .9rem;
  padding-left: .7rem;
  border-left: 3px solid var(--mixed);
}}
section {{ margin-bottom: 2.75rem; }}

.stat-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  border-radius: 10px;
  overflow: hidden;
}}
.stat {{
  background: var(--surface);
  padding: 1.1rem 1.2rem;
}}
.stat .n {{
  font-family: var(--font-mono);
  font-size: 1.6rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}}
.stat .n.bad {{ color: var(--bad); }}
.stat .n.good {{ color: var(--good); }}
.stat .l {{
  color: var(--muted);
  font-size: .78rem;
  margin-top: .2rem;
  line-height: 1.35;
}}

.finding {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 3px solid var(--bad);
  border-radius: 8px;
  padding: 1.1rem 1.3rem;
  margin-top: 1.1rem;
  font-size: .95rem;
  line-height: 1.55;
}}
.finding b {{ color: var(--bad); }}

.ds-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
}}
.ds-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 1rem 1.1rem 1.1rem;
}}
.ds-head {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: .5rem;
}}
.ds-name {{
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: .95rem;
}}
.ds-dims {{
  font-family: var(--font-mono);
  font-size: .72rem;
  color: var(--muted);
}}
.ds-charts {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: .5rem;
}}
.chart {{ min-height: 130px; }}
.chart svg {{ width: 100%; height: auto; overflow: visible; }}
.chart-title {{
  font-size: .68rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .04em;
  margin-bottom: .15rem;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  font-family: var(--font-mono);
  font-size: .8rem;
  font-variant-numeric: tabular-nums;
}}
.table-wrap {{
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--surface);
}}
th, td {{
  padding: .5rem .7rem;
  text-align: right;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}}
th {{
  text-align: right;
  font-family: var(--font-body);
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .04em;
  color: var(--muted);
  background: var(--surface-2);
  position: sticky;
  top: 0;
}}
td.c-ds, th:first-child {{ text-align: left; }}
tr.new-ds td {{ border-top: 2px solid var(--line); }}
td.bad {{ color: var(--bad); }}
td.good {{ color: var(--good); }}

footer {{
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: .8rem;
  line-height: 1.6;
}}
code {{ font-family: var(--font-mono); background: var(--surface-2); padding: .1em .35em; border-radius: 4px; font-size: .9em; }}
</style>

<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:wght@500;600&family=Source+Sans+3:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">

<div class="wrap">
<header>
  <h1>mp-kmeans on SuperKMeans: fp16_fp32 vs fp32</h1>
  <div class="subtitle">
    The authors&rsquo; own <code>mp-kmeans</code> package (arXiv:2407.12208,
    kernel <code>fp16_fp32</code>), run against its own <code>fp32</code>
    kernel from identical random initial centroids, on every vector-indexing
    dataset <code>scripts/fetch_superkmeans_datasets.sh</code> can produce.
  </div>
  <div class="meta-strip">
    <span>kernel <b>fp16_fp32</b></span>
    <span>kappa (&rho;) <b>1, 5</b></span>
    <span>k <b>32&ndash;1024</b></span>
    <span>iterations <b>fixed 50</b>, no early stop</span>
    <span>init <b>random</b>, same seed both kernels</span>
  </div>
  <div class="legend">
    <span><i class="swatch" style="background:var(--mixed)"></i>fp16_fp32 (mixed)</span>
    <span><i class="swatch" style="background:var(--ref)"></i>fp32 (reference, same package)</span>
  </div>
</header>

<section>
  <h2>At a glance</h2>
  <div class="stat-row">
    <div class="stat"><div class="n">{n_cells}</div><div class="l">(dataset, k, &kappa;) cells run</div></div>
    <div class="stat"><div class="n {'bad' if len(slower) > len(faster) else ''}">{len(slower)}/{n_cells}</div><div class="l">cells where fp16_fp32 ran <b>slower</b> than fp32</div></div>
    <div class="stat"><div class="n">{med_ratio:.2f}&times;</div><div class="l">median runtime ratio (mixed &divide; fp32)</div></div>
    <div class="stat"><div class="n bad">{worst['ratio']:.1f}&times;</div><div class="l">worst case: {esc(worst['dataset'])}, k={worst['k']}, &kappa;={worst['kappa']:g}</div></div>
  </div>
  <div class="finding">
    <b>The mixed kernel is not consistently a speedup.</b> Its cost is
    dominated by the reliability test&rsquo;s FP32 fallback path, which grows
    with k (more candidate centroids to re-check per point) &mdash; so at large k
    on real, non-separated embedding data it can run several times slower
    than the fp32 it&rsquo;s compared against, even though it&rsquo;s built
    around a cheaper FP16 GEMM. The charts below plot that ratio directly:
    above the 1&times; line is a loss, not a win.
  </div>
</section>

<section>
  <h2>Per dataset</h2>
  <div class="ds-grid">
  {''.join(dataset_cards)}
  </div>
</section>

<section>
  <h2>All cells</h2>
  <div class="table-wrap">
  <table>
    <thead><tr>
      <th>dataset</th><th>k</th><th>&kappa;</th><th>iters</th>
      <th>mixed time</th><th>fp32 time</th><th>ratio</th>
      <th>rel |&Delta;inertia|</th><th>labels moved</th>
    </tr></thead>
    <tbody>
    {''.join(rows_html)}
    </tbody>
  </table>
  </div>
</section>

<footer>
  Fixed 50 iterations, no convergence test, so runtime and inertia are
  comparable across cells without real data&rsquo;s convergence speed as a
  confound. &kappa;=1 and &kappa;=5 are the two "reliability factor" values the
  rest of this repo&rsquo;s evaluation uses (arXiv:2407.12208 prescribes 5;
  the measured knee for the exclusion test falls to ~1 by d&asymp;768). Rows
  with n over ~1.05M are subsampled &mdash; mp-kmeans&rsquo;s own CUDA kernels
  fail above that row count regardless of k or d, a limit found by bisection
  while building this sweep, not a memory ceiling.
</footer>
</div>

<script>
const DATA = {json.dumps(chart_data)};
const COLORS = {{1: getVar('--ref'), 5: getVar('--mixed')}};
function getVar(name) {{
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}}
function drawChart(el) {{
  const dataset = el.dataset.dataset, metric = el.dataset.metric;
  const series = DATA[dataset];
  const W = 260, H = 120, padL = 34, padR = 8, padT = 14, padB = 20;
  const ks = [...new Set(Object.values(series).flat().map(p => p.k))].sort((a,b)=>a-b);
  const xOf = k => padL + (Math.log2(k) - Math.log2(ks[0])) / (Math.log2(ks[ks.length-1]) - Math.log2(ks[0])) * (W - padL - padR);

  let allVals = [];
  for (const kap in series) for (const p of series[kap]) allVals.push(p[metric]);
  allVals = allVals.filter(v => isFinite(v));
  let lo = Math.min(...allVals), hi = Math.max(...allVals);
  const log = metric === 'rel_inertia';
  if (log) {{ lo = Math.max(lo, 1e-12); hi = Math.max(hi, lo * 10); lo = Math.log10(lo); hi = Math.log10(hi); }}
  else {{ const pad = (hi - lo) * 0.15 || 0.1; lo -= pad; hi += pad; if (metric === 'ratio') lo = Math.min(lo, 0.9); }}
  const yOf = v => {{
    const vv = log ? Math.log10(Math.max(v, 1e-12)) : v;
    return padT + (1 - (vv - lo) / (hi - lo || 1)) * (H - padT - padB);
  }};

  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}">`;
  const title = metric === 'ratio' ? 'runtime, mixed ÷ fp32' : 'relative |Δinertia|';
  svg += `<text x="0" y="10" font-size="8" fill="${{getVar('--muted')}}">${{title}}</text>`;

  // gridline + label at y=1 for ratio chart
  if (metric === 'ratio' && lo <= 1 && hi >= 1) {{
    const y1 = yOf(1);
    svg += `<line x1="${{padL}}" x2="${{W-padR}}" y1="${{y1}}" y2="${{y1}}" stroke="${{getVar('--line')}}" stroke-dasharray="2,2"/>`;
    svg += `<text x="${{padL-4}}" y="${{y1+2.5}}" font-size="7" text-anchor="end" fill="${{getVar('--muted')}}">1×</text>`;
  }}
  // y axis min/max labels
  svg += `<text x="${{padL-4}}" y="${{padT+2}}" font-size="7" text-anchor="end" fill="${{getVar('--muted')}}">${{log ? hi.toFixed(0) : hi.toFixed(2)}}${{log?'':''}}</text>`;
  svg += `<text x="${{padL-4}}" y="${{H-padB}}" font-size="7" text-anchor="end" fill="${{getVar('--muted')}}">${{log ? '10^'+lo.toFixed(0) : lo.toFixed(2)}}</text>`;

  // x axis ticks
  for (const k of ks) {{
    svg += `<text x="${{xOf(k)}}" y="${{H-4}}" font-size="7" text-anchor="middle" fill="${{getVar('--muted')}}">${{k}}</text>`;
  }}

  for (const kap of [1, 5]) {{
    const pts = (series[kap] || []).filter(p => isFinite(p[metric]));
    if (!pts.length) continue;
    const color = COLORS[kap];
    const path = pts.map((p,i) => `${{i===0?'M':'L'}}${{xOf(p.k).toFixed(1)}},${{yOf(p[metric]).toFixed(1)}}`).join(' ');
    svg += `<path d="${{path}}" fill="none" stroke="${{color}}" stroke-width="1.6"/>`;
    for (const p of pts) {{
      svg += `<circle cx="${{xOf(p.k).toFixed(1)}}" cy="${{yOf(p[metric]).toFixed(1)}}" r="2" fill="${{color}}"/>`;
    }}
  }}
  svg += `</svg>`;
  el.innerHTML = svg;
}}
document.querySelectorAll('.chart').forEach(drawChart);
</script>
"""
    with open(out_path, "w") as f:
        f.write(html)
    return n_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?",
                     default=os.path.join(os.path.dirname(__file__),
                                           "runs/super_kmeans_sweep/results.csv"))
    ap.add_argument("-o", "--out",
                     default=os.path.join(os.path.dirname(__file__),
                                           "runs/super_kmeans_sweep/report.html"))
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        sys.exit(f"no such file: {args.csv}")
    rows = load_rows(args.csv)
    if not rows:
        sys.exit(f"{args.csv}: no rt-mp-k* rows found")
    n = build(rows, args.out)
    print(f"wrote {args.out} ({n} cells, {len(set(r['dataset'] for r in rows))} datasets)")


if __name__ == "__main__":
    main()
