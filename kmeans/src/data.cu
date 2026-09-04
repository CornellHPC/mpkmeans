/* Synthetic data and centroid initialisation. */
#include "mpk_internal.cuh"

#include <random>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

extern "C" void mpkMakeBlobs(int n, int d, int k, float std, float center_box,
                             unsigned seed, float* hP, int* hLabels) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ubox(-center_box, center_box);
    std::normal_distribution<float> gauss(0.f, std);

    std::vector<float> centers((size_t)k * d);
    for (size_t t = 0; t < centers.size(); ++t) centers[t] = ubox(rng);

    for (int i = 0; i < n; ++i) {
        const int c = (int)((long long)i * k / n);   /* balanced clusters */
        if (hLabels) hLabels[i] = c;
        if (!hP) continue;
        for (int t = 0; t < d; ++t)
            hP[(size_t)i * d + t] = centers[(size_t)c * d + t] + gauss(rng);
    }
}

extern "C" mpkStatus mpkInitRandomPoints(const float* dP, int n, int d, int k,
                                         unsigned seed, float* dC) {
    if (k > n) return MPK_ERR_INVALID;
    std::mt19937 rng(seed ^ 0x9e3779b9u);

    /* partial Fisher-Yates over the index range, without materialising n ints
     * when n is large: reservoir-free selection via a hash set is overkill for
     * the sizes here, so just shuffle a vector. */
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> pick(i, n - 1);
        std::swap(idx[i], idx[pick(rng)]);
    }
    std::sort(idx.begin(), idx.begin() + k);

    for (int j = 0; j < k; ++j) {
        MPK_CUDA(cudaMemcpy(dC + (size_t)j * d, dP + (size_t)idx[j] * d,
                            (size_t)d * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    return MPK_OK;
}

/* ------------------------------------------------------------- libsvm --- */
/* LIBSVM / SVMlight sparse text format:
 *
 *     <label> <index>:<value> <index>:<value> ...
 *
 * Indices are 1-based.  Missing indices are zero, so the file is a sparse
 * encoding of a dense n x d matrix; d is the largest index that appears.  We
 * want the dense matrix -- the GEMM is dense either way, and the point of the
 * benchmark is the distance computation, not a sparse format.
 *
 * Two passes over the file: the first to learn n and d (nothing else can give
 * them), the second to fill.  That avoids holding the parsed triples in memory
 * as well as the dense matrix.
 *
 * The label column is kept as the ground truth for the agreement metric.  It
 * is whatever the file says -- class ids, +1/-1, or a regression target -- so
 * distinct values are numbered in order of first appearance, and the caller is
 * told how many there were.  It is never assumed to equal k.
 *
 * `qid:<n>` tokens (ranking datasets) are skipped, as is anything after '#'. */
extern "C" mpkStatus mpkLoadLibsvm(const char* path, int* out_n, int* out_d,
                                   float** out_P, int** out_labels,
                                   int* out_nclasses) {
    if (!path || !out_n || !out_d || !out_P) return MPK_ERR_INVALID;

    /* ---- pass 1: shape ------------------------------------------------- */
    long long rows = 0; long long maxidx = 0;
    {
        FILE* f = fopen(path, "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); return MPK_ERR_INVALID; }
        char* line = nullptr; size_t cap = 0; ssize_t len;
        while ((len = getline(&line, &cap, f)) != -1) {
            char* h = strchr(line, '#'); if (h) *h = '\0';
            char* p = line;
            while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
            if (!*p) continue;                       /* blank line */
            ++rows;
            for (char* c = p; (c = strchr(c, ':')) != nullptr; ++c) {
                /* walk back over the digits of the index */
                char* b = c; while (b > p && (b[-1] >= '0' && b[-1] <= '9')) --b;
                if (b == c) continue;                /* "qid:", not an index */
                const long long idx = strtoll(b, nullptr, 10);
                if (idx > maxidx) maxidx = idx;
            }
        }
        free(line); fclose(f);
    }
    if (rows == 0 || maxidx == 0) {
        fprintf(stderr, "%s: no usable rows (n=%lld, d=%lld)\n",
                path, rows, maxidx);
        return MPK_ERR_INVALID;
    }
    if (rows > INT_MAX || maxidx > INT_MAX ||
        (double)rows * (double)maxidx * sizeof(float) > 6e10) {
        fprintf(stderr, "%s: %lld x %lld is too large to densify\n",
                path, rows, maxidx);
        return MPK_ERR_INVALID;
    }
    const int n = (int)rows, d = (int)maxidx;

    /* ---- pass 2: fill --------------------------------------------------- */
    float* P = (float*)calloc((size_t)n * d, sizeof(float));
    int*   L = (int*)calloc((size_t)n, sizeof(int));
    if (!P || !L) { free(P); free(L); return MPK_ERR_ALLOC; }

    std::vector<double> classes;                     /* first-appearance order */
    {
        FILE* f = fopen(path, "r");
        if (!f) { free(P); free(L); return MPK_ERR_INVALID; }
        char* line = nullptr; size_t cap = 0; ssize_t len;
        long long i = 0;
        while ((len = getline(&line, &cap, f)) != -1) {
            char* h = strchr(line, '#'); if (h) *h = '\0';
            char* p = line;
            while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
            if (!*p) continue;

            char* end = nullptr;
            const double lab = strtod(p, &end);
            if (end == p) { fprintf(stderr, "%s:%lld: no label\n", path, i + 1); }
            p = end;
            while (*p && *p != ' ' && *p != '\t') ++p;   /* eat ",2" of a multilabel */

            size_t ci = 0;
            for (; ci < classes.size(); ++ci) if (classes[ci] == lab) break;
            if (ci == classes.size()) classes.push_back(lab);
            L[i] = (int)ci;

            float* row = P + (size_t)i * d;
            while (*p) {
                while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
                if (!*p) break;
                char* colon = nullptr;
                const long idx = strtol(p, &colon, 10);
                if (colon == p || *colon != ':') {           /* qid: or junk */
                    while (*p && *p != ' ' && *p != '\t') ++p;
                    continue;
                }
                char* vend = nullptr;
                const double v = strtod(colon + 1, &vend);
                if (idx >= 1 && idx <= d) row[idx - 1] = (float)v;
                p = (vend > colon + 1) ? vend : colon + 1;
            }
            ++i;
        }
        free(line); fclose(f);
    }

    *out_n = n; *out_d = d; *out_P = P;
    if (out_labels)   *out_labels = L; else free(L);
    if (out_nclasses) *out_nclasses = (int)classes.size();
    return MPK_OK;
}

/* Load a headerless float32 matrix -- the form SuperKMeans' setup_data.py
 * writes, and the form scripts/fetch_superkmeans_datasets.sh reproduces.
 *
 * There is nothing in the file but n*d float32 in row major order, so `d` has
 * to come from the caller (manifest.tsv next to the data records it) and n is
 * whatever the file size implies.  A size that is not a whole number of rows
 * means the dimension is wrong or the file is truncated, and is refused rather
 * than silently reinterpreted.
 *
 * These datasets carry no label column, so there is no ground truth to return
 * and the caller gets none -- unlike the LIBSVM path, where the label is the
 * first field of every row. */
extern "C" mpkStatus mpkLoadBin(const char* path, int d, long long max_rows,
                                int* out_n, float** out_P) {
    if (!path || d <= 0 || !out_n || !out_P) return MPK_ERR_INVALID;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return MPK_ERR_INVALID; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return MPK_ERR_INVALID; }
    const long long bytes = ftello(f);
    rewind(f);

    const long long row_bytes = (long long)d * (long long)sizeof(float);
    if (bytes <= 0 || bytes % row_bytes != 0) {
        fprintf(stderr, "%s: %lld bytes is not a whole number of %d-dim float32 "
                        "rows (%lld bytes each) -- wrong --dim, or truncated\n",
                path, bytes, d, row_bytes);
        fclose(f); return MPK_ERR_INVALID;
    }
    long long rows = bytes / row_bytes;
    if (max_rows > 0 && max_rows < rows) rows = max_rows;   /* prefix only */
    if (rows > INT_MAX) { fclose(f); return MPK_ERR_INVALID; }

    float* P = (float*)malloc((size_t)rows * d * sizeof(float));
    if (!P) { fclose(f); return MPK_ERR_ALLOC; }
    const size_t want = (size_t)rows * d;
    const size_t got  = fread(P, sizeof(float), want, f);
    fclose(f);
    if (got != want) {
        fprintf(stderr, "%s: read %zu of %zu floats\n", path, got, want);
        free(P); return MPK_ERR_INVALID;
    }
    *out_n = (int)rows;
    *out_P = P;
    return MPK_OK;
}
