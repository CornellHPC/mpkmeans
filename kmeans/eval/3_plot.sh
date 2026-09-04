#!/usr/bin/env bash
# (3/3) Turn the evaluation CSVs into plots.
#
# Bootstraps its own venv (numpy + matplotlib) the first time, then runs
# plot_results.py over every CSV under <runs>/results.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$HERE/runs"
OUT=""
VENV=""

usage() {
    cat <<USAGE
usage: $(basename "$0") [--runs DIR] [--out DIR] [--venv DIR]

  --runs DIR   the directory 2_gen_jobs.sh wrote (default $RUNS);
               CSVs are read from DIR/results
  --out DIR    where to write the plots (default DIR/plots)
  --venv DIR   python venv for numpy/matplotlib (default DIR/.plotvenv)
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -z "$OUT"  ]] && OUT="$RUNS/plots"
[[ -z "$VENV" ]] && VENV="$RUNS/.plotvenv"

[[ -d "$RUNS/results" ]] || {
    echo "no $RUNS/results -- run 2_gen_jobs.sh and then run_all.sh first" >&2
    exit 1
}

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "bootstrapping plotting venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" -q install --upgrade pip
    "$VENV/bin/pip" -q install numpy matplotlib
fi

exec "$VENV/bin/python" "$HERE/plot_results.py" --results "$RUNS/results" --out "$OUT"
