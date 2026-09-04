#!/usr/bin/env bash
# Download and prepare the SuperKMeans vector-indexing datasets.
#
#   https://github.com/cwida/SuperKMeans/blob/main/BENCHMARKING.md
#
# Those are real embedding corpora -- 200k to 2.3M vectors at 200 to 3072
# dimensions -- which is the regime this repo's exclusion bounds are supposed
# to pay off in and that gaussian blobs cannot stand in for.
#
# The nine datasets in that document's table live in a public Google Drive
# folder as .hdf5; the file ids below were read out of that folder on
# 2026-09-04 and are stable per file.  To re-read them:
#
#   gdown --folder -O /dev/null https://drive.google.com/drive/folders/$DRIVE_FOLDER
#
# What "prepare" means is fixed by their benchmarks/setup_data.py: take the
# `train` and `test` arrays out of the HDF5, cast to float32, L2-normalize the
# rows IF the dataset is on their angular list, and write them out as headerless
# row-major float32.  That is reproduced here rather than by calling their
# script, so that the conversion streams in row blocks instead of materialising
# a 7 GB array (arxiv) three times over.
#
#   NOTE the angular list is theirs, not the filename's: `openai-1536-angular`
#   is NOT normalized by their script, while `contriever-768` is.  Following the
#   names instead of the list would silently produce different data.
#
# Layout under --dest:
#
#   raw/          the .hdf5 exactly as downloaded (kept: re-converting is cheap,
#                 re-downloading 31 GB is not).  --drop-raw to discard.
#   data/         data_<id>.bin and data_<id>_test.bin, headerless float32.
#                 This is SuperKMeans' own `benchmarks/data/` layout, so their
#                 benchmarks can be pointed straight at it with a symlink.
#   manifest.tsv  id, shape, dtype, normalization and byte counts -- the .bin
#                 files carry no header, so nothing can read them without this.
#   README.md     the same, for a human.
#
# Idempotent: a dataset whose .bin files are already present at the right size
# is skipped, so an interrupted run can just be re-run.  Downloads resume.
set -euo pipefail

DRIVE_FOLDER=1f76UCrU52N2wToGMFg9ir1MY8ZocrN34
DEST=/global/homes/j/jbellav/m4646/hoon/mpkmeans-dsets
VENV=""
KEEP_RAW=1
FORCE=0
DRY=0

# id            drive_file_id                      hdf5_basename                      n        d     angular
# n and d are the documented shape (SuperKMeans benchmarks/bench_utils.py
# DATASET_PARAMS); the HDF5 is authoritative and a mismatch is reported, not
# enforced -- their own table and their own code disagree about contriever
# (999,000 vs 990,000), and the file is the tiebreaker.
DATASETS=(
  "arxiv      1k6FCpqQiUUXYo8J158dojQG1Bquz1yWg instructorxl-arxiv-768             2253000  768  0"
  "openai     1cwIY76n_HEbbZAANJVWiVkjBsIy3b9jB openai-1536-angular                 999000 1536  0"
  "jina       1q_cUqKbjpZ1jGijPMzWJunJ2Q_yjAHPZ codesearchnet-jina-768-cosine      1374067  768  1"
  "wiki       1730j9FT1q1Vn4BM1JghRzrtS90KlFV7e simplewiki-openai-3072-normalized   260372 3072  0"
  "mxbai      1SUHhKq0J_PmpRiwz7n7Wxcj4CS9-0qCf agnews-mxbai-1024-euclidean         769382 1024  0"
  "contriever 1rc5JDzR97S4CvVCjQ3pPI-MV9evVh7LO contriever-768                      990000  768  1"
  "clip       1nZC9Oun-A1_eR_K9F7I0MAXDbMHnZAKu imagenet-clip-512-normalized       1281167  512  0"
  "yahoo      1vtxoBOnJbIfqisW_5jA3mfemmcYEwDHc yahoo-minilm-384-normalized         677305  384  0"
  "yandex     1Dg6ZlLdKKU2nbYeZ9dA06vMQ8DqA8tpl yandex-200-cosine                  1000000  200  1"
  # not in BENCHMARKING.md's table, but mapped by their bench_utils.py and
  # present in the same folder -- available by name, never by default
  "sift       1PgRqbxYcImzhLboemhraX8sf5TRKm-_F sift-128-euclidean                 1000000  128  0"
  "gist       1q_iOZN6oGudT5lVv2hbn2awUKeV0npUR gist-960-euclidean                 1000000  960  0"
  "fmnist     17WTVbjWRQ9jyxeNBks8fJ_0KVHephqPb fashion-mnist-784-euclidean          60000  784  0"
  "glove200   1WYN55v5aNGrUwy0jmzAdSoxMyYhlRgx2 glove-200-angular                  1183514  200  1"
  "llama      1XBUhYJibKHfbRjYOr14XNnsS7mIyHIr6 llama-128-ip                        256921  128  1"
  "yi         1OCAiB41RazHe-p_zvPA52e8tVCp97mg3 yi-128-ip                           187843  128  1"
)
DEFAULT_IDS=(arxiv openai jina wiki mxbai contriever clip yahoo yandex)

usage() {
    cat <<EOF
usage: $(basename "$0") [options] [dataset ...]

Downloads and prepares the SuperKMeans vector-indexing datasets.
With no dataset named, does the nine from their BENCHMARKING.md table:
  ${DEFAULT_IDS[*]}
Also available by name (mapped by their bench_utils.py, same Drive folder):
  sift gist fmnist glove200 llama yi

options:
  --dest DIR    where to put everything   (default $DEST)
  --venv DIR    python venv for gdown/h5py/numpy  (default <dest>/.venv)
  --drop-raw    delete each .hdf5 once its .bin files exist (halves the
                footprint; re-preparing then means re-downloading)
  --force       re-download and re-convert even if outputs look complete
  --list        print the dataset table and exit
  --dry-run     say what would happen, download nothing
  -h, --help    this

The nine defaults are ~31 GB of HDF5 and ~31 GB of .bin, so budget ~62 GB
without --drop-raw.  cohere and openai5m are NOT here: they are HuggingFace
downloads (41 GB and 30 GB) rather than Drive files, and their retrieval lives
in SuperKMeans' own benchmarks/setup_data.py.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)     DEST="$2"; shift 2 ;;
        --venv)     VENV="$2"; shift 2 ;;
        --drop-raw) KEEP_RAW=0; shift ;;
        --force)    FORCE=1; shift ;;
        --dry-run)  DRY=1; shift ;;
        --list)
            printf "%-11s %-38s %10s %6s %s\n" id hdf5 n d normalized
            for row in "${DATASETS[@]}"; do
                read -r id _fid name n d ang <<<"$row"
                printf "%-11s %-38s %10s %6s %s\n" "$id" "$name.hdf5" "$n" "$d" \
                       "$([[ $ang == 1 ]] && echo yes || echo no)"
            done
            exit 0 ;;
        -h|--help)  usage; exit 0 ;;
        -*)         echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
        *)          break ;;
    esac
done

WANT=("$@")
[[ ${#WANT[@]} -eq 0 ]] && WANT=("${DEFAULT_IDS[@]}")

# validate names before doing anything slow
for want in "${WANT[@]}"; do
    found=0
    for row in "${DATASETS[@]}"; do
        read -r id _rest <<<"$row"
        [[ "$id" == "$want" ]] && found=1 && break
    done
    if [[ $found -eq 0 ]]; then
        echo "unknown dataset '$want'.  --list shows what is available." >&2
        case "$want" in
            cohere|cohere50m|openai5m)
                echo "  ('$want' is a HuggingFace download, not a Drive file --" >&2
                echo "   see SuperKMeans benchmarks/setup_data.py)" >&2 ;;
        esac
        exit 1
    fi
done

[[ -z "$VENV" ]] && VENV="$DEST/.venv"
RAW="$DEST/raw"
DATA="$DEST/data"

echo "dest      : $DEST"
echo "datasets  : ${WANT[*]}"
echo "raw hdf5  : $([[ $KEEP_RAW == 1 ]] && echo kept || echo "dropped after conversion")"
[[ $DRY == 1 ]] && echo "DRY RUN -- nothing will be downloaded or written"

if [[ $DRY == 0 ]]; then
    mkdir -p "$RAW" "$DATA"

    # --- venv -------------------------------------------------------------
    if [[ ! -x "$VENV/bin/python" ]]; then
        echo "bootstrapping venv at $VENV"
        python3 -m venv "$VENV"
        "$VENV/bin/pip" -q install --upgrade pip
        "$VENV/bin/pip" -q install gdown h5py numpy
    fi
    PY="$VENV/bin/python"
    GDOWN="$VENV/bin/gdown"
    "$PY" -c "import h5py, numpy, gdown" || {
        echo "venv at $VENV is missing gdown/h5py/numpy; delete it and re-run" >&2
        exit 1
    }

    # --- the converter, written once next to the venv ---------------------
    CONV="$VENV/superkmeans_prepare.py"
    cat > "$CONV" <<'PYEOF'
"""Convert one SuperKMeans .hdf5 into the headerless float32 .bin pair.

Mirrors SuperKMeans benchmarks/setup_data.py exactly -- cast to float32, then
for angular datasets divide each row by max(||row||_2, 1e-12) -- except that it
streams in row blocks rather than loading the whole array, and it reports the
shape it actually found instead of trusting the documented one.
"""
import sys, numpy as np, h5py

src, out_train, out_test, angular, doc_n, doc_d = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1",
    int(sys.argv[5]), int(sys.argv[6]))
BLOCK = 65536

def convert(dset, path):
    n, d = dset.shape
    with open(path, "wb") as f:
        for i in range(0, n, BLOCK):
            x = np.asarray(dset[i:i + BLOCK], dtype=np.float32)
            if angular:
                # float32 throughout, matching their l2_normalize
                nrm = np.linalg.norm(x, axis=1, keepdims=True)
                x = x / np.maximum(nrm, np.float32(1e-12))
            x.astype(np.float32, copy=False).tofile(f)
    return n, d

with h5py.File(src, "r") as h:
    for key in ("train", "test"):
        if key not in h:
            sys.exit(f"ERROR: {src} has no '{key}' dataset (has: {list(h.keys())})")
    n, d = convert(h["train"], out_train)
    tn, td = convert(h["test"], out_test)

if (n, d) != (doc_n, doc_d):
    print(f"  NOTE actual train shape ({n:,}, {d}) != documented "
          f"({doc_n:,}, {doc_d}); the file is authoritative")
if td != d:
    print(f"  WARNING test dim {td} != train dim {d}")
print(f"  train {n:,} x {d}   test {tn:,} x {td}   "
      f"{'L2-normalized' if angular else 'raw'}")
# machine readable tail for the manifest
print(f"MANIFEST\t{n}\t{d}\t{tn}\t{td}")
PYEOF
fi

# ---------------------------------------------------------------- main ----
declare -a MANIFEST_ROWS
for want in "${WANT[@]}"; do
    for row in "${DATASETS[@]}"; do
        read -r id fid name n d ang <<<"$row"
        [[ "$id" == "$want" ]] || continue

        hdf5="$RAW/$name.hdf5"
        train="$DATA/data_$id.bin"
        test="$DATA/data_${id}_test.bin"
        want_bytes=$(( n * d * 4 ))

        echo
        echo "=== $id  ($name.hdf5, documented ${n} x ${d}, $([[ $ang == 1 ]] && echo normalized || echo raw)) ==="

        if [[ $DRY == 1 ]]; then
            echo "  would download drive:$fid -> $hdf5"
            echo "  would write $train (~$(( want_bytes / 1000000 )) MB) and $test"
            continue
        fi

        # already done?
        if [[ $FORCE -eq 0 && -s "$train" && -s "$test" ]]; then
            have=$(stat -c %s "$train")
            if [[ "$have" -eq "$want_bytes" ]]; then
                echo "  already prepared ($train, $have bytes) -- skipping"
                continue
            fi
            echo "  $train is $have bytes, documented size is $want_bytes -- redoing"
        fi

        # download (resumes; --continue keeps a partial file across runs)
        if [[ $FORCE -eq 1 || ! -s "$hdf5" ]]; then
            echo "  downloading -> $hdf5"
            "$GDOWN" --no-cookies --continue -O "$hdf5" "$fid" || {
                echo "  FAILED to download $id." >&2
                echo "  Google Drive throttles large public files; if this says" >&2
                echo "  'quota exceeded', wait and re-run -- completed datasets" >&2
                echo "  are skipped, so progress is not lost.  The file can also" >&2
                echo "  be fetched by hand from" >&2
                echo "    https://drive.google.com/file/d/$fid/view" >&2
                echo "  and dropped at $hdf5." >&2
                exit 1
            }
        else
            echo "  raw already present ($hdf5)"
        fi

        echo "  converting"
        conv_out=$("$PY" "$CONV" "$hdf5" "$train" "$test" "$ang" "$n" "$d")
        echo "$conv_out" | grep -v '^MANIFEST' || true
        read -r _ an ad atn atd <<<"$(echo "$conv_out" | grep '^MANIFEST')"

        # the .bin has no header, so the size IS the shape check
        got=$(stat -c %s "$train")
        exp=$(( an * ad * 4 ))
        if [[ "$got" -ne "$exp" ]]; then
            echo "  ERROR $train is $got bytes, expected $exp for ${an} x ${ad}" >&2
            exit 1
        fi

        MANIFEST_ROWS+=("$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
            "$id" "$name" "$an" "$ad" "$atn" \
            "$([[ $ang == 1 ]] && echo l2 || echo none)" \
            "$got" "$(stat -c %s "$test")")")

        if [[ $KEEP_RAW -eq 0 ]]; then
            echo "  dropping $hdf5"
            rm -f "$hdf5"
        fi
    done
done

[[ $DRY == 1 ]] && exit 0

# ------------------------------------------------------------ manifest ----
MAN="$DEST/manifest.tsv"
if [[ ${#MANIFEST_ROWS[@]} -gt 0 ]]; then
    [[ -s "$MAN" ]] || printf 'id\thdf5\tn\td\ttest_n\tnormalized\ttrain_bytes\ttest_bytes\n' > "$MAN"
    for r in "${MANIFEST_ROWS[@]}"; do
        id=${r%%$'\t'*}
        grep -v "^${id}"$'\t' "$MAN" > "$MAN.tmp" || true
        mv "$MAN.tmp" "$MAN"
        printf '%s\n' "$r" >> "$MAN"
    done
    # keep the header first, rows sorted
    { head -1 "$MAN"; tail -n +2 "$MAN" | sort; } > "$MAN.tmp" && mv "$MAN.tmp" "$MAN"
fi

cat > "$DEST/README.md" <<EOF
# SuperKMeans vector-indexing datasets

Prepared by \`kmeans/scripts/fetch_superkmeans_datasets.sh\` from the Google
Drive folder linked in
<https://github.com/cwida/SuperKMeans/blob/main/BENCHMARKING.md>.

    raw/          the .hdf5 as downloaded (absent if --drop-raw was used)
    data/         data_<id>.bin, data_<id>_test.bin
    manifest.tsv  what each .bin actually contains

\`data/\` is laid out exactly as SuperKMeans' own \`benchmarks/data/\`, so their
benchmarks can use it directly:

    ln -s $DATA <superkmeans>/benchmarks/data

## Reading the .bin files

They are **headerless**: raw row-major float32, nothing else. The shape lives
only in \`manifest.tsv\`, so a reader has to be told it:

    import numpy as np
    x = np.fromfile("data/data_yandex.bin", dtype=np.float32).reshape(-1, 200)

Rows of the datasets marked \`l2\` in the manifest have been scaled to unit
length, following SuperKMeans' angular list -- which does not always match the
filename (\`openai-1536-angular\` is *not* normalized; \`contriever-768\` is).

Note this repo's own \`mpkmeans_bench --dataset\` reads LIBSVM text, not this
format, so it cannot consume these as they stand.

$(cat "$MAN" 2>/dev/null | column -t -s $'\t' | sed 's/^/    /')
EOF

echo
echo "done.  manifest:"
column -t -s $'\t' "$MAN"
echo
echo "wrote $DEST/README.md"
