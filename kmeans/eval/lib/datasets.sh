# Shared dataset table for the evaluation workflow.  Sourced by
# 1_prep_datasets.sh and 2_gen_jobs.sh so that the two cannot drift.
#
# LIBSVM rows:  id|url|bzipped?|filename|n|d
#   n and d are the documented shape, used only for the memory model and for a
#   post-download sanity check -- the loader reads whatever is in the file.
#   d is the LARGEST FEATURE INDEX, which is what mpkLoadLibsvm densifies to.
#
# BIN rows:     id|file|d
#   prepared by scripts/fetch_superkmeans_datasets.sh; d comes from its
#   manifest.tsv, and n from the file size.

LIBSVM_BASE=https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets

LIBSVM_SETS=(
  "covtype|${LIBSVM_BASE}/binary/covtype.libsvm.binary.scale.bz2|1|covtype.libsvm.binary.scale|581012|54"
  "realsim|${LIBSVM_BASE}/binary/real-sim.bz2|1|real-sim|72309|20958"
  "epsilon|${LIBSVM_BASE}/binary/epsilon_normalized.bz2|1|epsilon_normalized|400000|2000"
  "news20|${LIBSVM_BASE}/binary/news20.binary.bz2|1|news20.binary|19996|1355191"
  "wiki10|${LIBSVM_BASE}/multilabel/Wiki10-31K.bz2|1|Wiki10-31K|20762|101938"
  "mnist8m|${LIBSVM_BASE}/multiclass/mnist8m.scale.bz2|1|mnist8m.scale|8100000|784"
)

# the three named in instructions/eval.md; the other six the fetch script can
# produce are not part of this evaluation
BIN_SETS=(
  "openai|data_openai.bin|1536"
  "arxiv|data_arxiv.bin|768"
  "wiki|data_wiki.bin|3072"
)
BIN_FETCH_IDS="openai arxiv wiki"

# Device memory model for one mpkmeans_bench invocation, in bytes.
#
#   dP + dPraw                  8*n*d   (shifted and unshifted copies)
#   the two fp16 P planes       4*n*d   (multiword split; the widest driver)
#   G + G2buf                   8*n*k
#   per-row bookkeeping        ~44*n    (bestpack, jbest/dbest/gbest/gexact,
#                                        prev, two label arrays, survivor list)
#
# Measured against the real thing it is conservative by roughly 15%, which is
# the margin we want: it decides whether a job is generated at all.
mem_bytes() {  # n d k -> bytes  (bash int64 is enough: peak here is ~8e10)
    local n=$1 d=$2 k=$3
    echo $(( 12*n*d + 8*n*k + 44*n ))
}

# Largest n that fits a budget at the given d and k (0 if even one row will not)
max_n_for() {  # d k budget_bytes -> n
    local d=$1 k=$2 b=$3
    local per=$(( 12*d + 8*k + 44 ))
    echo $(( b / per ))
}

# mpkLoadLibsvm refuses to densify beyond 6e10 bytes, and that happens on the
# HOST before any subsampling, so it is a hard gate on n*d for LIBSVM sources.
LIBSVM_DENSE_LIMIT=60000000000

# The arXiv:2407.12208 baseline's error model is gamma_l(d+2) with u_l = 2^-11,
# which is only defined while (d+2)*u_l < 1.  At or past this d, mpkMeansBaselineRT
# declines the problem and the benchmark drops that one scheme -- the other
# seven still run, so the job is not lost, but the rt-base row will be absent.
RT_MAX_D=2045
