#!/usr/bin/env bash
# (1/3) Download and prepare every dataset the evaluation needs.
#
# Three sources, three shapes:
#
#   synthetic     nothing to download.  mpkmeans_bench generates blobs itself
#                 from -n/-d/-k/-s/-b, so this script only records that they
#                 exist.
#   LIBSVM        bz2 text, decompressed here.  mpkLoadLibsvm densifies to the
#                 largest feature index, so a sparse file with a huge index
#                 space becomes a huge dense matrix -- which is why the
#                 feasibility column below matters and why news20 cannot be run.
#   vector index  the SuperKMeans corpora, delegated to
#                 scripts/fetch_superkmeans_datasets.sh (Google Drive -> hdf5 ->
#                 headerless float32 + manifest).
#
# Everything lands under --dest, and 2_gen_jobs.sh reads the same --dest.
# Idempotent: anything already present at the right size is left alone.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/lib/datasets.sh"

DEST=/global/homes/j/jbellav/m4646/hoon/mpkmeans-dsets
ONLY=""
KEEP_BZ2=0

usage() {
    cat <<USAGE
usage: $(basename "$0") [--dest DIR] [--only libsvm|vector] [--keep-bz2]

Downloads and prepares the evaluation datasets into DIR (default $DEST):

  DIR/libsvm/     decompressed LIBSVM text files
  DIR/data/       headerless float32 vector-indexing matrices
  DIR/raw/        the .hdf5 those came from
  DIR/manifest.tsv, DIR/libsvm/manifest.tsv

Synthetic data needs no preparation -- mpkmeans_bench generates it.

Budget roughly 120 GB: ~66 GB for the vector corpora (kept with their .hdf5
sources) and ~50 GB for the decompressed LIBSVM text, most of it mnist8m.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --keep-bz2) KEEP_BZ2=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

LIBSVM_DIR="$DEST/libsvm"
mkdir -p "$LIBSVM_DIR"

# ------------------------------------------------------------- LIBSVM ----
if [[ -z "$ONLY" || "$ONLY" == libsvm ]]; then
    echo "=== LIBSVM datasets -> $LIBSVM_DIR ==="
    MAN="$LIBSVM_DIR/manifest.tsv"
    printf 'id\tfile\tdoc_n\tdoc_d\tdense_gb\tbytes\n' > "$MAN"

    for row in "${LIBSVM_SETS[@]}"; do
        IFS='|' read -r id url bz name n d <<<"$row"
        out="$LIBSVM_DIR/$name"
        gb=$(python3 -c "print('%.1f' % ($n*$d*4/1e9))")

        echo
        echo "--- $id  ($name, ${n} x ${d}, ${gb} GB dense) ---"

        if [[ -s "$out" ]]; then
            echo "  already present ($(stat -c %s "$out") bytes)"
        else
            arc="$LIBSVM_DIR/$(basename "$url")"
            if [[ ! -s "$arc" ]]; then
                echo "  downloading $url"
                # -c so an interrupted transfer resumes rather than restarts
                wget -q --show-progress -c -O "$arc" "$url" || {
                    echo "  FAILED to download $id from $url" >&2
                    echo "  (LIBSVM occasionally moves files; check the page at" >&2
                    echo "   $LIBSVM_BASE and update eval/lib/datasets.sh)" >&2
                    rm -f "$arc"; exit 1
                }
            else
                echo "  archive already present"
            fi
            if [[ "$bz" == 1 ]]; then
                echo "  decompressing (this is the slow part for mnist8m)"
                bunzip2 -kc "$arc" > "$out.part" && mv "$out.part" "$out"
                [[ $KEEP_BZ2 == 1 ]] || rm -f "$arc"
            else
                mv "$arc" "$out"
            fi
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$name" "$n" "$d" "$gb" \
               "$(stat -c %s "$out")" >> "$MAN"
    done
    echo
    column -t -s $'\t' "$MAN"
fi

# ------------------------------------------------ vector indexing sets ----
if [[ -z "$ONLY" || "$ONLY" == vector ]]; then
    echo
    echo "=== vector-indexing datasets -> $DEST/data ==="
    # shellcheck disable=SC2086
    "$REPO/scripts/fetch_superkmeans_datasets.sh" --dest "$DEST" $BIN_FETCH_IDS
fi

# ---------------------------------------------------------- synthetic ----
echo
echo "=== synthetic ==="
echo "  generated in-process by mpkmeans_bench (-n -d -k -s -b); nothing to download"

echo
echo "done.  point 2_gen_jobs.sh at --data $DEST"
