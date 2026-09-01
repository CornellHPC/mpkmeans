/* Batched cross products + a single reduction.
 *
 * One cublasGemmBatchedEx writes every product to its own slice of a
 * temporary workspace; one kernel then reduces those slices into the output.
 * Two launches regardless of how many products there are, which is what makes
 * the small-n shapes viable -- there the per-product Range drivers are
 * launch-bound rather than arithmetic-bound.
 *
 * Reduction layout: one thread per element of the REAL output. Each thread
 * walks the slices for its own (row, col), summing in a register. Consecutive
 * threads take consecutive rows, so each slice read is fully coalesced, and
 * single ownership of each output element means no atomics anywhere.
 */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

#include <cstdint>
#include <vector>

namespace {

constexpr int kBlock = 256;
constexpr size_t kAlign = 256;
inline size_t alignUp(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

/* FP32 slices, unit weights (bf16 and multiword fp16: the split magnitudes
 * already carry the bin scaling). */
__global__ void reduceF32(const float* __restrict__ T, int64_t ldt, int64_t strideT,
                          int batch, float alpha, float beta,
                          float* __restrict__ C, int64_t ldc,
                          int64_t rows, int64_t cols)
{
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    for (int64_t c = blockIdx.y; c < cols; c += gridDim.y) {
        const float* __restrict__ p = T + c * ldt + r;
        float s = 0.0f;
        for (int b = 0; b < batch; ++b) s += p[(int64_t)b * strideT];
        float* dst = C + c * ldc + r;
        *dst = (beta == 0.0f) ? alpha * s : beta * (*dst) + alpha * s;
    }
}

/* INT32 slices with the Ozaki I per-product scale folded in:
 * 2^(eA_i + eB_j - (bin_b + 2)*alphaBits). Accumulating in a register across
 * all products also avoids the repeated round-trips through C that the
 * per-product fold performs. */
__global__ void reduceI32Ozaki(const int* __restrict__ T, int64_t ldt, int64_t strideT,
                               int batch, const int* __restrict__ bins, int alphaBits,
                               const int* __restrict__ eA, const int* __restrict__ eB,
                               double alpha, double beta,
                               double* __restrict__ C, int64_t ldc,
                               int64_t rows, int64_t cols)
{
    extern __shared__ int sBins[];
    for (int i = threadIdx.x; i < batch; i += blockDim.x) sBins[i] = bins[i];
    __syncthreads();

    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int ea = eA[r];

    for (int64_t c = blockIdx.y; c < cols; c += gridDim.y) {
        const int eb = eB[c];
        double s = 0.0;
        if (ea != MPEMU_OZ1_ZERO_EXP && eb != MPEMU_OZ1_ZERO_EXP) {
            const int* __restrict__ p = T + c * ldt + r;
            const int base = ea + eb;
            for (int b = 0; b < batch; ++b)
                s += ldexp((double)p[(int64_t)b * strideT],
                           base - (sBins[b] + 2) * alphaBits);
        }
        double* dst = C + c * ldc + r;
        *dst = (beta == 0.0) ? alpha * s : beta * (*dst) + alpha * s;
    }
}

mpemuStatus_t launchBatchedGemm(cublasHandle_t handle,
                                cublasOperation_t opA, cublasOperation_t opB,
                                int64_t m, int64_t n, int64_t k,
                                const void* alpha, cudaDataType inType,
                                const mpemuBatchWorkspace_t* bw,
                                int planBegin, int planCount,
                                int64_t lda, int64_t ldb,
                                const void* beta, cudaDataType outType,
                                cublasComputeType_t compute)
{
    const cublasStatus_t st = cublasGemmBatchedEx(
        handle, opA, opB, (int)m, (int)n, (int)k,
        alpha,
        bw->aptr + planBegin, inType, (int)lda,
        bw->bptr + planBegin, inType, (int)ldb,
        beta,
        bw->cptr + planBegin, outType, (int)bw->ldt,
        planCount, compute, CUBLAS_GEMM_DEFAULT);
    return (st == CUBLAS_STATUS_SUCCESS) ? MPEMU_STATUS_SUCCESS
                                         : MPEMU_STATUS_CUBLAS_ERROR;
}

}  /* namespace */

extern "C" size_t mpemuBatchWorkspaceBytes(int64_t m, int64_t n, int capacity,
                                           int elemBytes)
{
    if (capacity < 1 || m < 0 || n < 0 || elemBytes < 1) return 0;
    const int64_t ldt = (m + 15) / 16 * 16;
    const size_t slices = (size_t)ldt * (size_t)n * (size_t)capacity * (size_t)elemBytes;
    const size_t ptrs   = (size_t)capacity * sizeof(void*);
    return alignUp(slices) + 3 * alignUp(ptrs) + alignUp((size_t)capacity * sizeof(int));
}

extern "C" mpemuStatus_t mpemuBatchWorkspaceInit(mpemuContext_t ctx,
                                                 int64_t m, int64_t n,
                                                 int capacity, int elemBytes,
                                                 size_t arenaOffset,
                                                 mpemuBatchWorkspace_t* bw)
{
    if (!ctx || !bw || capacity < 1 || elemBytes < 1) return MPEMU_STATUS_INVALID_VALUE;
    const size_t need = mpemuBatchWorkspaceBytes(m, n, capacity, elemBytes);
    const size_t start = alignUp(arenaOffset);
    if (start + need > mpemuContextCapacity(ctx)) return MPEMU_STATUS_INVALID_VALUE;

    char* arena = (char*)mpemuContextBase(ctx);
    if (!arena) return MPEMU_STATUS_INVALID_VALUE;
    char* base = arena + start;

    bw->ldt      = (m + 15) / 16 * 16;
    bw->strideT  = bw->ldt * n;
    bw->capacity = capacity;
    bw->planned  = 0;
    bw->elemBytes = elemBytes;

    size_t off = 0;
    bw->tmp  = base + off;
    off = alignUp(off + (size_t)bw->strideT * capacity * elemBytes);
    bw->aptr = (const void**)(base + off); off = alignUp(off + (size_t)capacity * sizeof(void*));
    bw->bptr = (const void**)(base + off); off = alignUp(off + (size_t)capacity * sizeof(void*));
    bw->cptr = (void**)      (base + off); off = alignUp(off + (size_t)capacity * sizeof(void*));
    bw->bins = (int*)        (base + off);
    return MPEMU_STATUS_SUCCESS;
}

extern "C" mpemuStatus_t mpemuBatchPlan(mpemuBatchWorkspace_t* bw,
                                        const void* dSA, int64_t strideA,
                                        const void* dSB, int64_t strideB,
                                        int splitElemBytes,
                                        int nsplits, int macBegin, int macEnd,
                                        cudaStream_t stream)
{
    if (!bw || !dSA || !dSB || splitElemBytes < 1) return MPEMU_STATUS_INVALID_VALUE;
    if (macEnd == MPEMU_ALL_MACS) macEnd = mpemuTermCount(nsplits);
    const int count = macEnd - macBegin;
    if (macBegin < 0 || count < 1 || count > bw->capacity) return MPEMU_STATUS_INVALID_VALUE;

    std::vector<mpemuTerm_t> terms(macEnd);
    if (mpemuTermSchedule(nsplits, macEnd, terms.data()) < macEnd)
        return MPEMU_STATUS_INVALID_VALUE;

    std::vector<const void*> ha(count), hb(count);
    std::vector<void*>       hc(count);
    std::vector<int>         hbin(count);
    for (int t = 0; t < count; ++t) {
        const mpemuTerm_t& tm = terms[macBegin + t];
        ha[t]   = (const char*)dSA + (size_t)tm.i * strideA * splitElemBytes;
        hb[t]   = (const char*)dSB + (size_t)tm.j * strideB * splitElemBytes;
        hc[t]   = (char*)bw->tmp   + (size_t)t    * bw->strideT * bw->elemBytes;
        hbin[t] = tm.i + tm.j;
    }
    const size_t pb = (size_t)count * sizeof(void*);
    if (cudaMemcpyAsync((void*)bw->aptr, ha.data(), pb, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync((void*)bw->bptr, hb.data(), pb, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync((void*)bw->cptr, hc.data(), pb, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(bw->bins, hbin.data(), (size_t)count * sizeof(int),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return MPEMU_STATUS_CUDA_ERROR;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return MPEMU_STATUS_CUDA_ERROR;

    bw->planned = count;
    return MPEMU_STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ */

static mpemuStatus_t batchedF32(cublasHandle_t handle,
                                cublasOperation_t opA, cublasOperation_t opB,
                                int64_t m, int64_t n, int64_t k,
                                float alpha, cudaDataType inType,
                                int planBegin, int planCount,
                                float beta, float* dC, int64_t ldc,
                                const mpemuBatchWorkspace_t* bw,
                                int64_t lda, int64_t ldb,
                                cudaStream_t stream)
{
    if (!bw || !dC) return MPEMU_STATUS_INVALID_VALUE;
    if (planBegin < 0 || planCount < 1 || planBegin + planCount > bw->planned)
        return MPEMU_STATUS_INVALID_VALUE;

    const float one = 1.0f, zero = 0.0f;
    const mpemuStatus_t st = launchBatchedGemm(handle, opA, opB, m, n, k,
        &one, inType, bw, planBegin, planCount, lda, ldb,
        &zero, CUDA_R_32F, CUBLAS_COMPUTE_32F);
    if (st != MPEMU_STATUS_SUCCESS) return st;

    dim3 grid((unsigned)((m + kBlock - 1) / kBlock),
              (unsigned)(n < 65535 ? n : 65535));
    reduceF32<<<grid, kBlock, 0, stream>>>(
        (const float*)bw->tmp + (int64_t)planBegin * bw->strideT,
        bw->ldt, bw->strideT, planCount, alpha, beta, dC, ldc, m, n);
    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}

extern "C" mpemuStatus_t mpemuGemmEmulatedBatched(cublasHandle_t handle,
                                                  cublasOperation_t opA, cublasOperation_t opB,
                                                  int64_t m, int64_t n, int64_t k,
                                                  float alpha, int planBegin, int planCount,
                                                  float beta, float* dC, int64_t ldc,
                                                  const mpemuBatchWorkspace_t* bw,
                                                  cudaStream_t stream)
{
    /* split planes: A is m x k with ld = mpemuSplitLd(m), B is k x n */
    return batchedF32(handle, opA, opB, m, n, k, alpha, CUDA_R_16BF,
                      planBegin, planCount, beta, dC, ldc, bw,
                      mpemuSplitLd(m), mpemuSplitLd(k), stream);
}

extern "C" mpemuStatus_t mpemuGemmMultiwordBatched(cublasHandle_t handle,
                                                   cublasOperation_t opA, cublasOperation_t opB,
                                                   int64_t m, int64_t n, int64_t k,
                                                   float alpha, float scaleA, float scaleB,
                                                   int planBegin, int planCount,
                                                   float beta, float* dC, int64_t ldc,
                                                   const mpemuBatchWorkspace_t* bw,
                                                   cudaStream_t stream)
{
    if (!(scaleA > 0.0f) || !(scaleB > 0.0f)) return MPEMU_STATUS_INVALID_VALUE;
    return batchedF32(handle, opA, opB, m, n, k, alpha / (scaleA * scaleB), CUDA_R_16F,
                      planBegin, planCount, beta, dC, ldc, bw,
                      mpemuSplitLd(m), mpemuSplitLd(k), stream);
}

extern "C" mpemuStatus_t mpemuGemmOzaki1Batched(cublasHandle_t handle,
                                                int64_t m, int64_t n, int64_t k,
                                                double alpha,
                                                const int* dExpA, const int* dExpB,
                                                int alphaBits,
                                                int planBegin, int planCount,
                                                double beta, double* dC, int64_t ldc,
                                                const mpemuBatchWorkspace_t* bw,
                                                cudaStream_t stream)
{
    if (!bw || !dC || !dExpA || !dExpB) return MPEMU_STATUS_INVALID_VALUE;
    if (planBegin < 0 || planCount < 1 || planBegin + planCount > bw->planned)
        return MPEMU_STATUS_INVALID_VALUE;
    if (alphaBits < 1) return MPEMU_STATUS_INVALID_VALUE;

    const int one = 1, zero = 0;
    /* TN is the only layout that reaches the IMMA fast path on A100 */
    const mpemuStatus_t st = launchBatchedGemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, k, &one, CUDA_R_8I, bw, planBegin, planCount,
        mpemuOz1SplitLd(k), mpemuOz1SplitLd(k),
        &zero, CUDA_R_32I, CUBLAS_COMPUTE_32I);
    if (st != MPEMU_STATUS_SUCCESS) return st;

    dim3 grid((unsigned)((m + kBlock - 1) / kBlock),
              (unsigned)(n < 65535 ? n : 65535));
    const size_t shmem = (size_t)planCount * sizeof(int);
    reduceI32Ozaki<<<grid, kBlock, shmem, stream>>>(
        (const int*)bw->tmp + (int64_t)planBegin * bw->strideT,
        bw->ldt, bw->strideT, planCount, bw->bins + planBegin, alphaBits,
        dExpA, dExpB, alpha, beta, dC, ldc, m, n);
    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}
