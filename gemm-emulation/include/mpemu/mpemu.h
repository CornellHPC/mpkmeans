/*
 * mpemu -- FP32 GEMM emulation on BF16 tensor cores.
 *
 * Implements the decomposition of Henry, Tang & Heinecke, "Leveraging the
 * bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations"
 * (ARITH 2019, arXiv:1904.06376).
 *
 * An FP32 value a is split into nsplits BF16 values by repeated
 * round-to-nearest conversion of the running residual:
 *
 *     b[0] = BF16(a)
 *     b[1] = BF16(a - b[0])
 *     b[2] = BF16(a - b[0] - b[1])
 *
 * Each subtraction is exact in FP32, so with nsplits=3 the representation
 * a == b[0] + b[1] + b[2] is exact for every normal FP32 input whose
 * exponent is not so small that b[1]/b[2] fall into the BF16 denormal range
 * (see MPEMU exponent-range note in the README).
 *
 * The product op(A)*op(B) is then assembled from cross products
 * op(A_i)*op(B_j), each computed by a BF16 tensor-core GEMM with FP32
 * accumulation. Terms are ordered by ascending bin i+j, so truncating the
 * term list after any number of terms drops the least significant
 * contributions first. For nsplits=3 the classical schemes are the first
 * 1, 3, 6 and 9 terms.
 *
 * Conventions: column-major, cuBLAS-native leading dimensions and
 * cublasOperation_t transpose flags. All matrices are device resident.
 */

#ifndef MPEMU_MPEMU_H
#define MPEMU_MPEMU_H

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#if defined(_WIN32)
#  define MPEMU_API __declspec(dllimport)
#else
#  define MPEMU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MPEMU_STATUS_SUCCESS       = 0,
    MPEMU_STATUS_INVALID_VALUE = 1,
    MPEMU_STATUS_CUDA_ERROR    = 2,
    MPEMU_STATUS_CUBLAS_ERROR  = 3
} mpemuStatus_t;

/* Word count for the BF16 / multiword-fp16 splits. Beyond 3 adds nothing for
 * FP32 input (the residual is already zero), but the code is written for a
 * general count. */
#define MPEMU_MAX_SPLITS 4

/* Ozaki scheme I uses many more slices than the mantissa-split schemes (8 for
 * FP64), and shares their cross-product ordering, so mpemuTermCount and
 * mpemuTermSchedule accept split counts up to this larger bound. */
#define MPEMU_MAX_TERM_SPLITS 16

MPEMU_API const char* mpemuStatusString(mpemuStatus_t s);

/* ------------------------------------------------------------------ */
/* Sizing helpers for the packed split buffer                          */
/* ------------------------------------------------------------------ */

/* Leading dimension mpemu uses for a split plane with `rows` rows: rows
 * rounded up to a multiple of 8 BF16 elements, which both keeps every column
 * 16-byte aligned for the vectorised split kernel and satisfies the alignment
 * cuBLAS wants for BF16 tensor-core GEMMs. */
MPEMU_API int64_t mpemuSplitLd(int64_t rows);

/* Element distance between consecutive split planes in the packed buffer. */
MPEMU_API int64_t mpemuSplitStride(int64_t lds, int64_t cols);

/* Bytes needed for a packed nsplits-plane buffer for a rows x cols matrix. */
MPEMU_API size_t mpemuSplitBytes(int64_t rows, int64_t cols, int nsplits);

/* Pointer to split plane s inside a packed buffer. */
static inline __nv_bfloat16* mpemuSplitPlane(__nv_bfloat16* dS,
                                             int64_t splitStride, int s)
{
    return dS + (int64_t)s * splitStride;
}
static inline const __nv_bfloat16* mpemuSplitPlaneConst(const __nv_bfloat16* dS,
                                                        int64_t splitStride, int s)
{
    return dS + (int64_t)s * splitStride;
}

/* ------------------------------------------------------------------ */
/* Routine 1: split an FP32 device matrix into nsplits BF16 planes      */
/* ------------------------------------------------------------------ */
/*
 * dA   [in]  device, rows x cols, column-major, leading dimension lda.
 * dS   [out] device, packed buffer of nsplits planes; plane s holds the
 *            s-th BF16 component, column-major with leading dimension lds
 *            and plane-to-plane element stride splitStride.
 *
 * Pass lds = mpemuSplitLd(rows) and splitStride = mpemuSplitStride(lds, cols)
 * for the layout mpemuSplitBytes sizes; any larger values are also accepted.
 * Runs entirely on the GPU, asynchronously on `stream`.
 */
MPEMU_API mpemuStatus_t mpemuSplitBF16(const float* dA, int64_t lda,
                                       int64_t rows, int64_t cols,
                                       int nsplits,
                                       __nv_bfloat16* dS,
                                       int64_t lds, int64_t splitStride,
                                       cudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Routine 2: one BF16 tensor-core cross product, FP32 accumulation     */
/* ------------------------------------------------------------------ */
/*
 * Computes  dC := alpha * op(dAi) * op(dBj) + beta * dC
 * via a single cublasGemmEx_64 with CUDA_R_16BF inputs, CUDA_R_32F output
 * and CUBLAS_COMPUTE_32F.
 *
 * dAi and dBj are individual split planes (from mpemuSplitPlane). dC is an
 * ordinary FP32 device matrix.
 *
 * Call with beta = 0.0f for the first term of a product and beta = 1.0f for
 * every subsequent term, which lets a caller add cross products one at a
 * time and inspect dC in between:
 *
 *     mpemuTerm_t terms[9];
 *     int nterms = mpemuTermSchedule(3, 9, terms);
 *     int done = 0;
 *     while (need_more_accuracy(...)) {
 *         int target = done + batch;
 *         for (; done < target && done < nterms; ++done)
 *             mpemuGemmBF16(h, opA, opB, m, n, k, 1.0f,
 *                           mpemuSplitPlaneConst(dSA, strideA, terms[done].i), lda,
 *                           mpemuSplitPlaneConst(dSB, strideB, terms[done].j), ldb,
 *                           done == 0 ? 0.0f : 1.0f, dC, ldc);
 *     }
 *
 * The routine keeps no state: the number of cross products accumulated is
 * entirely the caller's choice and is independent of the split count.
 */
MPEMU_API mpemuStatus_t mpemuGemmBF16(cublasHandle_t handle,
                                      cublasOperation_t opA, cublasOperation_t opB,
                                      int64_t m, int64_t n, int64_t k,
                                      float alpha,
                                      const __nv_bfloat16* dAi, int64_t lda,
                                      const __nv_bfloat16* dBj, int64_t ldb,
                                      float beta,
                                      float* dC, int64_t ldc);

/* ------------------------------------------------------------------ */
/* Cross-product ordering                                              */
/* ------------------------------------------------------------------ */

typedef struct { int i; int j; } mpemuTerm_t;

/* Total number of cross products available for nsplits: nsplits^2.
 * Accepts nsplits up to MPEMU_MAX_TERM_SPLITS. */
MPEMU_API int mpemuTermCount(int nsplits);

/* Fill `terms` with up to maxTerms cross products ordered by ascending bin
 * (i+j), and by ascending i within a bin -- i.e. most significant
 * contribution first. Returns the number written. For nsplits=3 the first
 * 1/3/6/9 entries are the paper's 1-, 3-, 6- and 9-product schemes.
 * `terms` must have room for min(maxTerms, nsplits^2) entries. */
MPEMU_API int mpemuTermSchedule(int nsplits, int maxTerms, mpemuTerm_t* terms);

/* ------------------------------------------------------------------ */
/* Convenience drivers over routines 1 and 2                           */
/* ------------------------------------------------------------------ */

/* For macEnd: "every remaining term in the schedule". */
#define MPEMU_ALL_MACS (-1)

/*
 * Accumulate the cross products in schedule positions [macBegin, macEnd)
 * into an existing FP32 result:
 *
 *     dC := alpha * sum_{t in [macBegin,macEnd)} op(A_it) * op(B_jt) + beta * dC
 *
 * This is the incremental-refinement entry point. Split the operands once,
 * spend `c` multiply-accumulates, inspect dC, and then extend the SAME dC
 * with more terms -- without recomputing what you already have:
 *
 *     // first pass: c terms, starting from scratch
 *     mpemuGemmEmulatedRange(h, opA, opB, m, n, k, 1.0f,
 *                            dSA, ldsA, strA, dSB, ldsB, strB,
 *                            nsplits, 0, c, 0.0f, dC, ldc);
 *
 *     while (!accurate_enough(dC) && c < mpemuTermCount(nsplits)) {
 *         int next = c + d;                       // a few more products
 *         mpemuGemmEmulatedRange(h, opA, opB, m, n, k, 1.0f,
 *                                dSA, ldsA, strA, dSB, ldsB, strB,
 *                                nsplits, c, next, 1.0f, dC, ldc);
 *         c = next;
 *     }
 *
 *     // or just finish the job:
 *     mpemuGemmEmulatedRange(..., nsplits, c, MPEMU_ALL_MACS, 1.0f, dC, ldc);
 *
 * Refining is exact, not approximate: because each term is an independent
 * GEMM accumulating into dC, splitting the work as [0,c) then [c,e) issues
 * the identical sequence of cuBLAS calls as a single [0,e), so the result is
 * BITWISE identical to computing e terms in one go. mpemu_test_refine
 * asserts this.
 *
 * `beta` scales dC once, before the first term of THIS call -- pass 0.0f to
 * start a fresh result and 1.0f to extend one. macEnd = MPEMU_ALL_MACS means
 * mpemuTermCount(nsplits). An empty range just applies beta.
 */
MPEMU_API mpemuStatus_t mpemuGemmEmulatedRange(cublasHandle_t handle,
                                               cublasOperation_t opA, cublasOperation_t opB,
                                               int64_t m, int64_t n, int64_t k,
                                               float alpha,
                                               const __nv_bfloat16* dSA, int64_t lda, int64_t strideA,
                                               const __nv_bfloat16* dSB, int64_t ldb, int64_t strideB,
                                               int nsplits, int macBegin, int macEnd,
                                               float beta,
                                               float* dC, int64_t ldc);

/*
 * Accumulates the first `nmacs` cross products of the schedule:
 *     dC := alpha * op(A) * op(B) + beta * dC
 * Exactly mpemuGemmEmulatedRange(..., 0, nmacs, ...); provided for callers
 * that know their MAC budget up front. `beta` is applied once, on the first
 * term.
 */
MPEMU_API mpemuStatus_t mpemuGemmEmulated(cublasHandle_t handle,
                                          cublasOperation_t opA, cublasOperation_t opB,
                                          int64_t m, int64_t n, int64_t k,
                                          float alpha,
                                          const __nv_bfloat16* dSA, int64_t lda, int64_t strideA,
                                          const __nv_bfloat16* dSB, int64_t ldb, int64_t strideB,
                                          int nsplits, int nmacs,
                                          float beta,
                                          float* dC, int64_t ldc);

/* ==================================================================== */
/* Multiword fp16 arithmetic                                            */
/* ==================================================================== */
/*
 * M. Fasi, N. J. Higham, F. Lopez, T. Mary, M. Mikaitis, "Matrix
 * Multiplication in Multiword Arithmetic: Error Analysis and Application to
 * GPU Tensor Cores", MIMS EPrint 2022.3 (presented at GAMM 2022). A
 * generalization of the Markidis et al. (2018) precision-refinement idea.
 *
 * The split has the same recursive form as the BF16 scheme above, but into
 * `p` fp16 words:
 *
 *     A_i = fl_fp16( A - sum_{k<i} A_k )
 *
 * The truncation rule differs. Rather than "take the first k cross products",
 * the paper drops every product with i + j > p + 1 (1-based), leaving
 * p(p+1)/2 products instead of p^2. In the ascending-bin ordering used by
 * mpemuTermSchedule that set is exactly the first p(p+1)/2 terms, so the same
 * schedule applies; mpemuMultiwordMacs(p) returns that count.
 *
 * For fp16 (u_low = 2^-11) against fp32 accumulation (u_high = 2^-24) the
 * paper shows p = 2 is sufficient, because u_low^2 = 4*u_high. That is the
 * headline "double-fp16" scheme:
 *
 *     AB ~= A1*B1 + A1*B2 + A2*B1        (three tensor-core GEMMs)
 *
 * with error bound n*2^-24 -- i.e. fp32-class accuracy at fp16 throughput.
 * p = 3 does NOT help for fp16: the third word lands at ~2^-23 relative,
 * which is subnormal in fp16, so it carries almost no information.
 *
 * EXPONENT RANGE. fp16's smallest normal is 6.104e-5, far narrower than
 * bf16's. The analysis assumes "no exceptions occur in the conversion", but
 * for A ~ U(-0.5, 0.5] the second word already lands near 2.4e-5 and is
 * subnormal. mpemuSplitFP16 therefore takes a `scale` argument: pass 1.0f for
 * the literal scheme, or a power of two (see mpemuAutoScaleFP16) to move the
 * words into fp16's normal range. Scaling by a power of two is exact, so it
 * introduces no error of its own; the caller divides it back out through
 * alpha. mpemuCheckSplitFP16 reports how many words actually overflowed or
 * went subnormal.
 */

/* Number of cross products the paper recommends for a p-word split:
 * p(p+1)/2, i.e. every (i,j) with i + j <= p + 1 in 1-based indexing.
 * p=1 -> 1, p=2 -> 3 (double-fp16), p=3 -> 6. */
MPEMU_API int mpemuMultiwordMacs(int p);

/* ------------------------------------------------------------------ */
/* Routine 1 (fp16): split an FP32 device matrix into p fp16 words     */
/* ------------------------------------------------------------------ */
/*
 * Identical in shape to mpemuSplitBF16, with one extra argument.
 *
 * scale [in] multiplies every element before splitting. Pass 1.0f for the
 *            scheme exactly as published. Pass a power of two to keep the
 *            words inside fp16's normal range; the product then carries a
 *            factor scaleA*scaleB that the caller removes via alpha (the
 *            mpemuGemmMultiword driver does this for you).
 *
 * The packed-buffer layout, and therefore mpemuSplitLd / mpemuSplitStride /
 * mpemuSplitBytes, is shared with the BF16 routines: fp16 and bf16 are both
 * 2 bytes, so the sizing is identical.
 */
MPEMU_API mpemuStatus_t mpemuSplitFP16(const float* dA, int64_t lda,
                                       int64_t rows, int64_t cols,
                                       int nsplits, float scale,
                                       __half* dS,
                                       int64_t lds, int64_t splitStride,
                                       cudaStream_t stream);

static inline __half* mpemuSplitPlaneFP16(__half* dS, int64_t splitStride, int s)
{
    return dS + (int64_t)s * splitStride;
}
static inline const __half* mpemuSplitPlaneFP16Const(const __half* dS,
                                                     int64_t splitStride, int s)
{
    return dS + (int64_t)s * splitStride;
}

/* Largest power of two `scale` such that scale*max|A| stays safely inside
 * fp16's normal range, giving the trailing words as much exponent room as
 * possible. Writes the scale to *hScale (host) and synchronises `stream`.
 * Returns scale = 1 for an all-zero matrix.
 *
 * Uses one cached device scratch word and synchronises `stream`, so it is a
 * setup call rather than a hot path, and concurrent calls from several host
 * threads must be serialised. */
MPEMU_API mpemuStatus_t mpemuAutoScaleFP16(const float* dA, int64_t lda,
                                           int64_t rows, int64_t cols,
                                           float* hScale, cudaStream_t stream);

/* Diagnostic for the paper's "no exceptions" precondition. counts is a device
 * array of 3 uint64: [0] words that overflowed to +-inf, [1] words that were
 * subnormal or flushed to zero while the residual was still nonzero, and
 * [2] words that converted cleanly. Counts across all p planes. */
MPEMU_API mpemuStatus_t mpemuCheckSplitFP16(const float* dA, int64_t lda,
                                            int64_t rows, int64_t cols,
                                            int nsplits, float scale,
                                            const __half* dS,
                                            int64_t lds, int64_t splitStride,
                                            unsigned long long* dCounts,
                                            cudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Routine 2 (fp16): one fp16 cross product, FP32 accumulation         */
/* ------------------------------------------------------------------ */
/*
 * dC := alpha * op(dAi) * op(dBj) + beta * dC, as a single cublasGemmEx_64
 * with CUDA_R_16F inputs, CUDA_R_32F output and CUBLAS_COMPUTE_32F.
 *
 * Stateless, exactly like mpemuGemmBF16: pass beta = 0 for the first term and
 * beta = 1 to refine, and accumulate as many cross products as you like.
 */
MPEMU_API mpemuStatus_t mpemuGemmFP16(cublasHandle_t handle,
                                      cublasOperation_t opA, cublasOperation_t opB,
                                      int64_t m, int64_t n, int64_t k,
                                      float alpha,
                                      const __half* dAi, int64_t lda,
                                      const __half* dBj, int64_t ldb,
                                      float beta,
                                      float* dC, int64_t ldc);

/* ------------------------------------------------------------------ */
/* Convenience driver                                                  */
/* ------------------------------------------------------------------ */
/*
 * Accumulates the first `nmacs` cross products in ascending-bin order.
 * Pass nmacs = mpemuMultiwordMacs(p) for the paper's recommended scheme
 * (p = 2, nmacs = 3 is double-fp16).
 *
 * scaleA / scaleB are the values handed to mpemuSplitFP16; the driver folds
 * 1/(scaleA*scaleB) into alpha so dC comes back in the original units. Pass
 * 1.0f for both if the operands were split unscaled.
 */
/*
 * Incremental form, mirroring mpemuGemmEmulatedRange: accumulate schedule
 * positions [macBegin, macEnd) into an existing dC. Same bitwise-resumability
 * guarantee. macEnd = MPEMU_ALL_MACS means every remaining term.
 *
 * Note the multiword schedule is ordered so that position p(p+1)/2 is the
 * paper's recommended cut; refining past mpemuMultiwordMacs(p) adds the
 * i+j > p+1 terms, which measurably change nothing (see RESULTS_MULTIWORD.md).
 */
MPEMU_API mpemuStatus_t mpemuGemmMultiwordRange(cublasHandle_t handle,
                                                cublasOperation_t opA, cublasOperation_t opB,
                                                int64_t m, int64_t n, int64_t k,
                                                float alpha,
                                                const __half* dSA, int64_t lda, int64_t strideA,
                                                float scaleA,
                                                const __half* dSB, int64_t ldb, int64_t strideB,
                                                float scaleB,
                                                int nsplits, int macBegin, int macEnd,
                                                float beta,
                                                float* dC, int64_t ldc);

MPEMU_API mpemuStatus_t mpemuGemmMultiword(cublasHandle_t handle,
                                           cublasOperation_t opA, cublasOperation_t opB,
                                           int64_t m, int64_t n, int64_t k,
                                           float alpha,
                                           const __half* dSA, int64_t lda, int64_t strideA,
                                           float scaleA,
                                           const __half* dSB, int64_t ldb, int64_t strideB,
                                           float scaleB,
                                           int nsplits, int nmacs,
                                           float beta,
                                           float* dC, int64_t ldc);

/* ==================================================================== */
/* Ozaki scheme I -- FP64 GEMM on INT8 tensor cores (ozIMMU)            */
/* ==================================================================== */
/*
 * H. Ootomo, K. Ozaki, R. Yokota, "DGEMM on Integer Matrix Multiplication
 * Unit", arXiv:2306.11975. Algorithms 3 and 4 (the integer variant).
 *
 * Not a floating-point mantissa split: each row of A (column of B) gets one
 * SHARED power-of-two exponent, and the mantissas are then cut into fixed-
 * point slices of `alpha` bits held in INT8. Because INT8 tensor cores
 * accumulate in INT32, every slice product A^(i) B^(j) is computed with NO
 * rounding error at all, provided
 *
 *     alpha = min( floor((31 - log2 k) / 2), 7 )
 *
 * which is what mpemuOz1BitsPerSlice returns: 2*alpha bits for the product
 * plus log2(k) bits of accumulation must fit INT32's 31 magnitude bits.
 * For k <= 2^17 this gives the full alpha = 7 bits per slice.
 *
 * The result is reassembled in FP64 as (Algorithm 3, line 7)
 *
 *     C += (A^(i) B^(j)) * 2^-((i+j)*alpha) * e_A (x) e_B
 *
 * Accuracy is set by the slice count s; s = 8 covers FP64's 53 bits.
 *
 * REFINEMENT. This scheme is the most refinement-friendly of those in this
 * library, because nothing about a slice depends on how many slices you
 * eventually ask for:
 *   - alpha depends only on the accumulator width and k;
 *   - the shared exponent e is computed once from the matrix;
 *   - slice p is an independent bit-field, so extending s -> s+1 leaves
 *     slices 1..s bit-identical and merely appends one.
 * The product set {i+j <= s+1} is exactly the first s(s+1)/2 entries of the
 * ascending-bin order that mpemuTermSchedule already produces, so growing s
 * only ADDS terms. Use mpemuGemmOzaki1Range to spend more of them later.
 *
 * Note the cost of refining is quadratic: going from s to s+1 slices adds
 * s+1 new matrix products, not one.
 */

/* Bits per slice: min(floor((31 - ceil(log2 k)) / 2), 7). */
MPEMU_API int mpemuOz1BitsPerSlice(int64_t k);

/* Matrix products for s slices: s(s+1)/2, i.e. every (i,j) with i+j <= s+1. */
MPEMU_API int mpemuOz1Macs(int s);

/* Largest inner dimension for which one slice product cannot overflow INT32. */
MPEMU_API int64_t mpemuOz1MaxK(int alpha);

/* Slice planes are stored so the INT8 GEMM runs in its fast TN form: for both
 * operands the planes have k rows, and the shared exponent is indexed by the
 * plane's COLUMN. Pass leftOperand = 1 for A (m x k, transposed into k x m)
 * and 0 for B (k x n, copied as is). */
MPEMU_API int64_t mpemuOz1SplitLd(int64_t rows);
MPEMU_API size_t  mpemuOz1SplitBytes(int64_t planeRows, int64_t planeCols, int s);

/* Shared exponents: dExp[c] = floor(log2 max|.|) + 1 over the elements that
 * land in plane column c, so every scaled magnitude is strictly below 1.
 * An all-zero row/column yields MPEMU_OZ1_ZERO_EXP. */
#define MPEMU_OZ1_ZERO_EXP (-30000)

MPEMU_API mpemuStatus_t mpemuOz1Exponents(const double* dM, int64_t ld,
                                          int64_t rows, int64_t cols,
                                          int leftOperand,
                                          int* dExp, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Routine 1: FP64 matrix -> s INT8 fixed-point slice planes           */
/* ------------------------------------------------------------------ */
/*
 * Slice p holds mantissa bits [(p-1)*alpha, p*alpha - 1] of |M| / 2^e, with
 * the sign of M. Each slice therefore lies in [-(2^alpha - 1), 2^alpha - 1],
 * which fits INT8 for alpha <= 7.
 *
 * Slices are independent bit-fields, so calling this with a larger `nslices`
 * reproduces the earlier planes exactly -- that is what makes refinement
 * cheap. Requires nslices * alpha <= 63 (the extraction uses a 64-bit
 * fixed-point word); with alpha = 7 that allows up to 9 slices, and FP64's
 * 53 significand bits are covered by 8.
 */
MPEMU_API mpemuStatus_t mpemuSplitInt8Ozaki1(const double* dM, int64_t ld,
                                             int64_t rows, int64_t cols,
                                             int leftOperand,
                                             int nslices, int alpha,
                                             const int* dExp,
                                             signed char* dS,
                                             int64_t lds, int64_t splitStride,
                                             cudaStream_t stream);

static inline signed char* mpemuOz1Plane(signed char* dS, int64_t stride, int p)
{ return dS + (int64_t)p * stride; }
static inline const signed char* mpemuOz1PlaneConst(const signed char* dS,
                                                    int64_t stride, int p)
{ return dS + (int64_t)p * stride; }

/* ------------------------------------------------------------------ */
/* Routine 2: one INT8 tensor-core GEMM with INT32 accumulation        */
/* ------------------------------------------------------------------ */
/*
 * dC := alpha * op(dAi) * op(dBj) + beta * dC, one cublasGemmEx_64 with
 * CUDA_R_8I inputs, CUDA_R_32I output and CUBLAS_COMPUTE_32I. Exact: no
 * rounding occurs at all while INT32 does not overflow.
 *
 * On A100 only the TN form (opA = CUBLAS_OP_T, opB = CUBLAS_OP_N) reaches the
 * IMMA fast path -- measured 427 TOPS against 61-67 for the other three
 * layouts -- which is why routine 1 stores the left operand transposed.
 *
 * Stateless, like mpemuGemmBF16 / mpemuGemmFP16. Pass beta = 1 to sum several
 * slice products of the SAME bin i+j into one INT32 buffer before converting.
 */
MPEMU_API mpemuStatus_t mpemuGemmInt8(cublasHandle_t handle,
                                      cublasOperation_t opA, cublasOperation_t opB,
                                      int64_t m, int64_t n, int64_t k,
                                      int alpha,
                                      const signed char* dAi, int64_t lda,
                                      const signed char* dBj, int64_t ldb,
                                      int beta,
                                      int* dC, int64_t ldc);

/* ------------------------------------------------------------------ */
/* Folding INT32 slice products back into the FP64 result              */
/* ------------------------------------------------------------------ */
/*
 * dC := alpha * 2^-((bin+2)*alphaBits) * 2^(eA_i + eB_j) * dCt + beta * dC
 *
 * `bin` is the zero-based i+j of the slice pair, so the 1-based (i+j) of the
 * paper is bin + 2. Products sharing a bin share this scale, which is why the
 * driver accumulates a whole bin in INT32 and converts once.
 */
MPEMU_API mpemuStatus_t mpemuOz1Accumulate(const int* dCt, int64_t ldct,
                                           int64_t rows, int64_t cols,
                                           int bin, int alphaBits,
                                           const int* dExpA, const int* dExpB,
                                           double alpha, double beta,
                                           double* dC, int64_t ldc,
                                           cudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Drivers                                                             */
/* ------------------------------------------------------------------ */
/*
 * Accumulate schedule positions [macBegin, macEnd) into an existing FP64 dC,
 * mirroring mpemuGemmEmulatedRange. dCt is an m x n INT32 scratch buffer.
 *
 * Because slices are independent bit-fields and the product set only grows
 * with s, this is a genuine refinement: split once with the largest s you may
 * want, spend s(s+1)/2 products for a small s, inspect dC, then extend.
 * mpemuOz1Macs(s) gives the cut corresponding to exactly s slices.
 */
MPEMU_API mpemuStatus_t mpemuGemmOzaki1Range(cublasHandle_t handle,
                                             int64_t m, int64_t n, int64_t k,
                                             double alpha,
                                             const signed char* dSA, int64_t ldsa, int64_t strideA,
                                             const int* dExpA,
                                             const signed char* dSB, int64_t ldsb, int64_t strideB,
                                             const int* dExpB,
                                             int nslices, int alphaBits,
                                             int macBegin, int macEnd,
                                             double beta,
                                             double* dC, int64_t ldc,
                                             int* dCt, int64_t ldct,
                                             cudaStream_t stream);

MPEMU_API mpemuStatus_t mpemuGemmOzaki1(cublasHandle_t handle,
                                        int64_t m, int64_t n, int64_t k,
                                        double alpha,
                                        const signed char* dSA, int64_t ldsa, int64_t strideA,
                                        const int* dExpA,
                                        const signed char* dSB, int64_t ldsb, int64_t strideB,
                                        const int* dExpB,
                                        int nslices, int alphaBits,
                                        double beta,
                                        double* dC, int64_t ldc,
                                        int* dCt, int64_t ldct,
                                        cudaStream_t stream);

/* ==================================================================== */
/* Context: allocate every workspace buffer once                        */
/* ==================================================================== */
/*
 * The routines above never allocate: split planes, INT32 scratch and exponent
 * vectors are all caller-owned. That is the right contract for a library, but
 * it pushes the bookkeeping onto the caller, and calling cudaMalloc/cudaFree
 * around a GEMM costs more than the GEMM for small operands.
 *
 * mpemuContext_t owns one growable device arena. Workspace views are carved
 * out of it with no allocation at all, so a steady-state loop -- split B,
 * accumulate products, repeat -- performs zero CUDA allocations.
 *
 *     mpemuContext_t ctx;
 *     mpemuContextCreate(&ctx);
 *     mpemuContextReserve(ctx, mpemuOz1WorkspaceBytes(m, n, k, 8));
 *
 *     mpemuOz1Workspace_t ws;
 *     mpemuOz1WorkspaceInit(ctx, m, n, k, 8, &ws);   // pointers only
 *
 *     // A is fixed: split it once, outside the loop
 *     mpemuOz1Exponents(dA, lda, m, k, 1, ws.expA, 0);
 *     mpemuSplitInt8Ozaki1(dA, lda, m, k, 1, 8, ws.alphaBits, ws.expA,
 *                          ws.sA, ws.ldA, ws.strideA, 0);
 *     for (...) {                                    // B changes each step
 *         mpemuOz1Exponents(dB, ldb, k, n, 0, ws.expB, 0);
 *         mpemuSplitInt8Ozaki1(dB, ldb, k, n, 0, 8, ws.alphaBits, ws.expB,
 *                              ws.sB, ws.ldB, ws.strideB, 0);
 *         mpemuGemmOzaki1(h, m, n, k, 1.0, ws.sA, ws.ldA, ws.strideA, ws.expA,
 *                         ws.sB, ws.ldB, ws.strideB, ws.expB, 8, ws.alphaBits,
 *                         0.0, dC, ldc, ws.ct, ws.ldct, 0);
 *     }
 *
 * Reserve is idempotent: it grows the arena when needed and is a no-op
 * otherwise, so it is safe to call every iteration.
 */

typedef struct mpemuContext_s* mpemuContext_t;

MPEMU_API mpemuStatus_t mpemuContextCreate(mpemuContext_t* ctx);
MPEMU_API mpemuStatus_t mpemuContextDestroy(mpemuContext_t ctx);

/* Grow the arena to at least `bytes`. No-op when it already fits. */
MPEMU_API mpemuStatus_t mpemuContextReserve(mpemuContext_t ctx, size_t bytes);
MPEMU_API size_t        mpemuContextCapacity(mpemuContext_t ctx);
/* Base of the arena, for callers carving their own views. NULL before the
 * first successful Reserve. */
MPEMU_API void*         mpemuContextBase(mpemuContext_t ctx);
/* Number of device allocations the context has performed, for asserting that
 * a steady-state loop is allocation-free. */
MPEMU_API int           mpemuContextAllocCount(mpemuContext_t ctx);

/* Arena bytes each scheme needs for one m x n x k problem. */
MPEMU_API size_t mpemuBf16WorkspaceBytes(int64_t m, int64_t n, int64_t k, int nsplits);
MPEMU_API size_t mpemuFp16WorkspaceBytes(int64_t m, int64_t n, int64_t k, int nsplits);
MPEMU_API size_t mpemuOz1WorkspaceBytes (int64_t m, int64_t n, int64_t k, int nslices);

typedef struct {
    __nv_bfloat16* sA; int64_t ldA, strideA;
    __nv_bfloat16* sB; int64_t ldB, strideB;
    int nsplits;
} mpemuBf16Workspace_t;

typedef struct {
    __half* sA; int64_t ldA, strideA;
    __half* sB; int64_t ldB, strideB;
    int nsplits;
} mpemuFp16Workspace_t;

typedef struct {
    signed char* sA; int64_t ldA, strideA;
    signed char* sB; int64_t ldB, strideB;
    int* expA;                 /* m entries */
    int* expB;                 /* n entries */
    int* ct;  int64_t ldct;    /* m x n INT32 product scratch */
    int  nslices, alphaBits;
    int  pad_;
} mpemuOz1Workspace_t;

/* Carve typed views out of the context arena. These allocate nothing; the
 * context must already have been reserved to at least the matching
 * *WorkspaceBytes, otherwise MPEMU_STATUS_INVALID_VALUE is returned. */
MPEMU_API mpemuStatus_t mpemuBf16WorkspaceInit(mpemuContext_t ctx,
                                               int64_t m, int64_t n, int64_t k,
                                               int nsplits,
                                               mpemuBf16Workspace_t* ws);
MPEMU_API mpemuStatus_t mpemuFp16WorkspaceInit(mpemuContext_t ctx,
                                               int64_t m, int64_t n, int64_t k,
                                               int nsplits,
                                               mpemuFp16Workspace_t* ws);
MPEMU_API mpemuStatus_t mpemuOz1WorkspaceInit(mpemuContext_t ctx,
                                              int64_t m, int64_t n, int64_t k,
                                              int nslices,
                                              mpemuOz1Workspace_t* ws);

/* ==================================================================== */
/* Batched variants: one GEMM launch + one reduction                    */
/* ==================================================================== */
/*
 * The Range drivers above issue one cuBLAS call per cross product, each
 * accumulating into C with beta=1. That is ideal for refinement, but on small
 * shapes the launches dominate: at 16384x64x100 the whole GEMM is ~21 us, so
 * 6 launches cost more than the arithmetic.
 *
 * These variants instead issue ALL the products as a single
 * cublasGemmBatchedEx into a temporary workspace -- one slice per product --
 * and then reduce that workspace into the output with a single kernel. Two
 * launches total, independent of the product count.
 *
 * The reduction assigns one thread per element of the real output; each
 * thread walks the temporary slices for its own (row, col) and sums them in a
 * register. Consecutive threads take consecutive rows, so every slice read is
 * coalesced, and because each output element is owned by exactly one thread
 * there are no atomics. For Ozaki I the per-product scale
 * 2^-((i+j+2)*alpha) * 2^(eA+eB) is applied inside that same pass, which also
 * removes the separate INT32->FP64 fold entirely.
 *
 * COST: the workspace is `capacity` slices of m x n. That is fine for the
 * small-n shapes this is meant for (36 slices of 16384x256 INT32 is 604 MB)
 * but explodes on large square problems (36 slices of 16384x16384 would be
 * 39 GB). Use the Range drivers there -- they are already GEMM-bound, so
 * batching buys nothing.
 *
 * These are additions: the per-product Range drivers are unchanged.
 */

typedef struct {
    void*        tmp;        /* capacity slices, ldt x n each */
    int64_t      ldt, strideT;
    const void** aptr;       /* device pointer arrays for cublasGemmBatchedEx */
    const void** bptr;
    void**       cptr;
    int*         bins;       /* device: i+j of each planned product */
    int          capacity;   /* slices the workspace holds */
    int          planned;    /* products currently planned */
    int          elemBytes;  /* 4: FP32 for bf16/fp16, INT32 for Ozaki I */
} mpemuBatchWorkspace_t;

/* Arena bytes for `capacity` temporary slices of an m x n output. */
MPEMU_API size_t mpemuBatchWorkspaceBytes(int64_t m, int64_t n, int capacity,
                                          int elemBytes);

/* Carve the batch workspace from a context (allocates nothing). Reserve the
 * context to at least mpemuBatchWorkspaceBytes + whatever the scheme's own
 * workspace needs. */
MPEMU_API mpemuStatus_t mpemuBatchWorkspaceInit(mpemuContext_t ctx,
                                                int64_t m, int64_t n,
                                                int capacity, int elemBytes,
                                                size_t arenaOffset,
                                                mpemuBatchWorkspace_t* bw);

/* Fill the device pointer arrays for schedule positions [macBegin, macEnd).
 * Depends only on the split BASE pointers and the schedule, both of which are
 * fixed for the lifetime of a workspace -- so plan once, outside the loop,
 * even though the split contents change every iteration.
 *
 * splitElemBytes is the size of one split-plane element: 2 for bf16 and fp16,
 * 1 for Ozaki I's INT8 slices. */
MPEMU_API mpemuStatus_t mpemuBatchPlan(mpemuBatchWorkspace_t* bw,
                                       const void* dSA, int64_t strideA,
                                       const void* dSB, int64_t strideB,
                                       int splitElemBytes,
                                       int nsplits, int macBegin, int macEnd,
                                       cudaStream_t stream);

/* dC := alpha * sum(planned products) + beta * dC, in two launches.
 * planBegin/planCount select a sub-range of the plan, so refinement in chunks
 * still works; pass 0 and bw->planned for everything. */
MPEMU_API mpemuStatus_t mpemuGemmEmulatedBatched(cublasHandle_t handle,
                                                 cublasOperation_t opA, cublasOperation_t opB,
                                                 int64_t m, int64_t n, int64_t k,
                                                 float alpha, int planBegin, int planCount,
                                                 float beta, float* dC, int64_t ldc,
                                                 const mpemuBatchWorkspace_t* bw,
                                                 cudaStream_t stream);

MPEMU_API mpemuStatus_t mpemuGemmMultiwordBatched(cublasHandle_t handle,
                                                  cublasOperation_t opA, cublasOperation_t opB,
                                                  int64_t m, int64_t n, int64_t k,
                                                  float alpha, float scaleA, float scaleB,
                                                  int planBegin, int planCount,
                                                  float beta, float* dC, int64_t ldc,
                                                  const mpemuBatchWorkspace_t* bw,
                                                  cudaStream_t stream);

/* Applies the Ozaki I scaling during the reduction, so no separate fold. */
MPEMU_API mpemuStatus_t mpemuGemmOzaki1Batched(cublasHandle_t handle,
                                               int64_t m, int64_t n, int64_t k,
                                               double alpha,
                                               const int* dExpA, const int* dExpB,
                                               int alphaBits,
                                               int planBegin, int planCount,
                                               double beta, double* dC, int64_t ldc,
                                               const mpemuBatchWorkspace_t* bw,
                                               cudaStream_t stream);

/* ==================================================================== */
/* Concatenated-k: one GEMM per bin, no temporary, no reduction         */
/* ==================================================================== */
/*
 * Every product in bin b = i+j carries the same weight, so
 *
 *     sum_{i=0..b} A_i B_{b-i}
 *
 * is a single GEMM of inner dimension (b+1)*k -- provided the planes sit
 * contiguously along k. No temporary workspace and no reduction pass at all,
 * which is what makes this beat the batched variant on memory-bound shapes.
 *
 * The LEFT operand already has the right layout: mpemuSplitBF16 writes plane i
 * at element offset i*strideA with strideA = lda*k, so viewing dSA as an
 * m x (nsplits*k) column-major matrix puts plane i at column block i*k.
 *
 * The RIGHT operand needs the planes stacked along rows and REVERSED --
 * plane j at row block (nsplits-1-j)*k. Then bin b is exactly
 *
 *     op(A)[:, 0 : (b+1)k]  *  Bstack[(nsplits-1-b)*k : nsplits*k, :]
 *
 * because A's block i sits at inner offset i*k while B's block (b-i) sits at
 * (nsplits-1-b+i)*k -- the two differ by the constant (nsplits-1-b)*k, so
 * sliding B's base by that amount pairs block i with block b-i for every i at
 * once. Verified in mpemu_test_concat.
 *
 * Cost: nsplits GEMMs for the full nsplits(nsplits+1)/2 products, and nsplits
 * accumulations into C rather than one per product. Same arithmetic.
 *
 * The right operand is the one that changes every iteration in the target
 * workload, and it is the small one, so re-splitting it into this layout costs
 * no more than the ordinary split.
 */

/* Leading dimension and byte size of a stacked right-operand buffer. */
MPEMU_API int64_t mpemuStackedLd(int64_t k, int nsplits);
MPEMU_API size_t  mpemuStackedBytes(int64_t k, int64_t n, int nsplits, int elemBytes);

/* Split the right operand directly into the stacked reversed layout. */
MPEMU_API mpemuStatus_t mpemuSplitBF16Stacked(const float* dB, int64_t ldb,
                                              int64_t k, int64_t n, int nsplits,
                                              __nv_bfloat16* dS, int64_t lds,
                                              cudaStream_t stream);
MPEMU_API mpemuStatus_t mpemuSplitFP16Stacked(const float* dB, int64_t ldb,
                                              int64_t k, int64_t n, int nsplits,
                                              float scale,
                                              __half* dS, int64_t lds,
                                              cudaStream_t stream);

/* dC := alpha * sum(products in bins [0, nbins)) + beta * dC.
 * nbins = nsplits covers the full nsplits(nsplits+1)/2 product set; a smaller
 * nbins is the same prefix the Range drivers would accumulate, so this still
 * refines -- just at bin granularity rather than per product. */
MPEMU_API mpemuStatus_t mpemuGemmEmulatedConcat(cublasHandle_t handle,
                                                int64_t m, int64_t n, int64_t k,
                                                float alpha,
                                                const __nv_bfloat16* dSA, int64_t lda,
                                                const __nv_bfloat16* dSB, int64_t ldsb,
                                                int nsplits, int nbins,
                                                float beta, float* dC, int64_t ldc);

MPEMU_API mpemuStatus_t mpemuGemmMultiwordConcat(cublasHandle_t handle,
                                                 int64_t m, int64_t n, int64_t k,
                                                 float alpha, float scaleA, float scaleB,
                                                 const __half* dSA, int64_t lda,
                                                 const __half* dSB, int64_t ldsb,
                                                 int nsplits, int nbins,
                                                 float beta, float* dC, int64_t ldc);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* MPEMU_MPEMU_H */
