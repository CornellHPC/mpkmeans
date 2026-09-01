/* Ozaki scheme I on INT8 tensor cores (ozIMMU): shared-exponent fixed-point
 * splitting, Algorithms 3 and 4 of Ootomo, Ozaki & Yokota (arXiv:2306.11975).
 *
 * Routine 1 (the split), the shared-exponent reduction it needs, and the
 * FP64 fold-back of INT32 slice products.
 */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

#include <cstdint>

namespace {

constexpr int kBlock = 256;
constexpr int kTile  = 32;   /* transpose tile */

/* ---------------------------------------------------------------- alpha -- */
/* 2*alpha bits of product plus log2(k) bits of accumulation must fit the 31
 * magnitude bits of INT32; INT8 inputs cap alpha at 7. */
int bitsPerSlice(int64_t k)
{
    int lg = 0;
    while (((int64_t)1 << lg) < k) ++lg;          /* ceil(log2 k) */
    int a = (31 - lg) / 2;
    if (a > 7) a = 7;
    if (a < 1) a = 0;
    return a;
}

/* ------------------------------------------------------- shared exponent -- */
/* dExp[c] = floor(log2 max|.|) + 1 over the elements mapping to plane column
 * c, so that |M| / 2^exp is strictly less than 1.
 *
 * leftOperand: the operand is A (rows x cols = m x k) and plane column c is
 * A's ROW c, so we reduce along cols. Otherwise the operand is B (k x n) and
 * plane column c is B's column c, so we reduce along rows. */
__global__ void expKernelRows(const double* __restrict__ M, int64_t ld,
                              int64_t rows, int64_t cols, int* __restrict__ e)
{
    /* one block per output column c = row index of A */
    for (int64_t c = blockIdx.x; c < rows; c += gridDim.x) {
        double best = 0.0;
        for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
            const double v = fabs(M[j * ld + c]);
            if (v > best && isfinite(v)) best = v;
        }
        __shared__ double sm[kBlock];
        sm[threadIdx.x] = best;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x])
                sm[threadIdx.x] = sm[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const double mx = sm[0];
            e[c] = (mx > 0.0) ? (int)floor(log2(mx)) + 1 : MPEMU_OZ1_ZERO_EXP;
        }
        __syncthreads();
    }
}

__global__ void expKernelCols(const double* __restrict__ M, int64_t ld,
                              int64_t rows, int64_t cols, int* __restrict__ e)
{
    /* one block per output column c = column index of B */
    for (int64_t c = blockIdx.x; c < cols; c += gridDim.x) {
        const double* __restrict__ col = M + c * ld;
        double best = 0.0;
        for (int64_t i = threadIdx.x; i < rows; i += blockDim.x) {
            const double v = fabs(col[i]);
            if (v > best && isfinite(v)) best = v;
        }
        __shared__ double sm[kBlock];
        sm[threadIdx.x] = best;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s && sm[threadIdx.x + s] > sm[threadIdx.x])
                sm[threadIdx.x] = sm[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const double mx = sm[0];
            e[c] = (mx > 0.0) ? (int)floor(log2(mx)) + 1 : MPEMU_OZ1_ZERO_EXP;
        }
        __syncthreads();
    }
}

/* ---------------------------------------------------------------- split -- */
/* Extract all slices of one value into a 64-bit fixed-point word.
 *
 * u = |v| / 2^e lies in [0,1), so F = floor(u * 2^total) fits in `total`
 * bits. Slice p is then the alpha-bit field starting at bit
 * total - p*alpha. Every slice is an independent bit-field of the same F,
 * which is precisely why extending the slice count leaves earlier planes
 * untouched. */
__device__ __forceinline__ unsigned long long fixedPoint(double v, int e, int total)
{
    if (e == MPEMU_OZ1_ZERO_EXP) return 0ull;
    const double u = fabs(v) * exp2((double)(-e));      /* exact: power of two */
    if (!(u > 0.0)) return 0ull;
    return (unsigned long long)(u * exp2((double)total));
}

/* B side: planes are a straight copy of B (k x n), exponent per column. */
__global__ void splitDirect(const double* __restrict__ M, int64_t ld,
                            int64_t rows, int64_t cols,
                            int nslices, int alpha, int total,
                            const int* __restrict__ e,
                            signed char* __restrict__ S,
                            int64_t lds, int64_t stride)
{
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const unsigned mask = (1u << alpha) - 1u;

    for (int64_t c = blockIdx.y; c < cols; c += gridDim.y) {
        const double v = M[c * ld + r];
        const unsigned long long F = fixedPoint(v, e[c], total);
        const int sign = (v < 0.0) ? -1 : 1;
#pragma unroll 1
        for (int p = 0; p < nslices; ++p) {
            const int shift = total - (p + 1) * alpha;
            const int d = (int)((F >> shift) & mask);
            S[p * stride + c * lds + r] = (signed char)(sign * d);
        }
    }
}

/* A side: planes are A transposed (k x m), exponent per plane column = A row.
 * Tiled so both the read of A and the write of the plane stay coalesced. */
__global__ void splitTransposed(const double* __restrict__ M, int64_t ld,
                                int64_t rows, int64_t cols,   /* A is rows x cols */
                                int nslices, int alpha, int total,
                                const int* __restrict__ e,
                                signed char* __restrict__ S,
                                int64_t lds, int64_t stride)
{
    __shared__ unsigned long long sF[kTile][kTile + 1];
    __shared__ int                sSign[kTile][kTile + 1];

    const int64_t r0 = (int64_t)blockIdx.x * kTile;   /* A row block   */
    const int64_t c0 = (int64_t)blockIdx.y * kTile;   /* A col block   */

    /* load a tile of A, converting to fixed point on the way in */
    for (int t = threadIdx.y; t < kTile; t += blockDim.y) {
        const int64_t c = c0 + t;
        const int64_t r = r0 + threadIdx.x;
        unsigned long long F = 0ull; int sg = 1;
        if (r < rows && c < cols) {
            const double v = M[c * ld + r];
            F  = fixedPoint(v, e[r], total);
            sg = (v < 0.0) ? -1 : 1;
        }
        sF[t][threadIdx.x] = F;
        sSign[t][threadIdx.x] = sg;
    }
    __syncthreads();

    /* write the transposed tile: plane element (l, i) with l = A col, i = A row */
    const unsigned mask = (1u << alpha) - 1u;
    for (int t = threadIdx.y; t < kTile; t += blockDim.y) {
        const int64_t i = r0 + t;                 /* plane column */
        const int64_t l = c0 + threadIdx.x;       /* plane row    */
        if (i >= rows || l >= cols) continue;
        const unsigned long long F = sF[threadIdx.x][t];
        const int sg = sSign[threadIdx.x][t];
#pragma unroll 1
        for (int p = 0; p < nslices; ++p) {
            const int shift = total - (p + 1) * alpha;
            const int d = (int)((F >> shift) & mask);
            S[p * stride + i * lds + l] = (signed char)(sg * d);
        }
    }
}

/* ----------------------------------------------------------- accumulate -- */
/* dC := alpha * 2^(eA_i + eB_j - (bin+2)*alphaBits) * dCt + beta * dC */
__global__ void accumKernel(const int* __restrict__ Ct, int64_t ldct,
                            int64_t rows, int64_t cols,
                            int shift, const int* __restrict__ eA,
                            const int* __restrict__ eB,
                            double alpha, double beta,
                            double* __restrict__ C, int64_t ldc)
{
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int ea = eA[r];

    for (int64_t c = blockIdx.y; c < cols; c += gridDim.y) {
        const int eb = eB[c];
        double add = 0.0;
        if (ea != MPEMU_OZ1_ZERO_EXP && eb != MPEMU_OZ1_ZERO_EXP) {
            const double t = (double)Ct[c * ldct + r];
            add = alpha * ldexp(t, ea + eb - shift);
        }
        double* dst = C + c * ldc + r;
        *dst = (beta == 0.0) ? add : beta * (*dst) + add;
    }
}

}  /* namespace */

extern "C" int mpemuOz1BitsPerSlice(int64_t k) { return bitsPerSlice(k); }

extern "C" int mpemuOz1Macs(int s)
{
    if (s < 1) return 0;
    return s * (s + 1) / 2;
}

extern "C" int64_t mpemuOz1MaxK(int alpha)
{
    if (alpha < 1 || alpha > 7) return 0;
    const int64_t d = ((int64_t)1 << alpha) - 1;
    return (int64_t)2147483647 / (d * d);
}

/* INT8 GEMM wants 16-byte-aligned leading dimensions. */
extern "C" int64_t mpemuOz1SplitLd(int64_t rows) { return (rows + 15) / 16 * 16; }

extern "C" size_t mpemuOz1SplitBytes(int64_t planeRows, int64_t planeCols, int s)
{
    if (s < 1) return 0;
    return (size_t)mpemuOz1SplitLd(planeRows) * (size_t)planeCols * (size_t)s;
}

extern "C" mpemuStatus_t mpemuOz1Exponents(const double* dM, int64_t ld,
                                           int64_t rows, int64_t cols,
                                           int leftOperand,
                                           int* dExp, cudaStream_t stream)
{
    if (!dM || !dExp) return MPEMU_STATUS_INVALID_VALUE;
    if (rows <= 0 || cols <= 0) return MPEMU_STATUS_SUCCESS;
    if (ld < rows) return MPEMU_STATUS_INVALID_VALUE;

    const int64_t nOut = leftOperand ? rows : cols;
    const unsigned grid = (unsigned)(nOut < 65535 ? nOut : 65535);
    if (leftOperand) expKernelRows<<<grid, kBlock, 0, stream>>>(dM, ld, rows, cols, dExp);
    else             expKernelCols<<<grid, kBlock, 0, stream>>>(dM, ld, rows, cols, dExp);

    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}

extern "C" mpemuStatus_t mpemuSplitInt8Ozaki1(const double* dM, int64_t ld,
                                              int64_t rows, int64_t cols,
                                              int leftOperand,
                                              int nslices, int alpha,
                                              const int* dExp,
                                              signed char* dS,
                                              int64_t lds, int64_t splitStride,
                                              cudaStream_t stream)
{
    if (!dM || !dS || !dExp) return MPEMU_STATUS_INVALID_VALUE;
    if (rows < 0 || cols < 0) return MPEMU_STATUS_INVALID_VALUE;
    if (nslices < 1 || alpha < 1 || alpha > 7) return MPEMU_STATUS_INVALID_VALUE;
    /* the bit-field extraction lives in one 64-bit word */
    const int total = nslices * alpha;
    if (total > 63) return MPEMU_STATUS_INVALID_VALUE;
    if (ld < rows) return MPEMU_STATUS_INVALID_VALUE;
    if (rows == 0 || cols == 0) return MPEMU_STATUS_SUCCESS;

    const int64_t planeRows = leftOperand ? cols : rows;
    const int64_t planeCols = leftOperand ? rows : cols;
    if (lds < planeRows) return MPEMU_STATUS_INVALID_VALUE;
    if (splitStride < lds * planeCols) return MPEMU_STATUS_INVALID_VALUE;

    if (leftOperand) {
        dim3 blk(kTile, 8);
        dim3 grid((unsigned)((rows + kTile - 1) / kTile),
                  (unsigned)((cols + kTile - 1) / kTile));
        splitTransposed<<<grid, blk, 0, stream>>>(dM, ld, rows, cols,
                                                  nslices, alpha, total, dExp,
                                                  dS, lds, splitStride);
    } else {
        dim3 grid((unsigned)((rows + kBlock - 1) / kBlock),
                  (unsigned)(cols < 65535 ? cols : 65535));
        splitDirect<<<grid, kBlock, 0, stream>>>(dM, ld, rows, cols,
                                                 nslices, alpha, total, dExp,
                                                 dS, lds, splitStride);
    }
    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}

extern "C" mpemuStatus_t mpemuOz1Accumulate(const int* dCt, int64_t ldct,
                                            int64_t rows, int64_t cols,
                                            int bin, int alphaBits,
                                            const int* dExpA, const int* dExpB,
                                            double alpha, double beta,
                                            double* dC, int64_t ldc,
                                            cudaStream_t stream)
{
    if (!dCt || !dC || !dExpA || !dExpB) return MPEMU_STATUS_INVALID_VALUE;
    if (rows <= 0 || cols <= 0) return MPEMU_STATUS_SUCCESS;
    if (bin < 0 || alphaBits < 1) return MPEMU_STATUS_INVALID_VALUE;

    const int shift = (bin + 2) * alphaBits;
    dim3 grid((unsigned)((rows + kBlock - 1) / kBlock),
              (unsigned)(cols < 65535 ? cols : 65535));
    accumKernel<<<grid, kBlock, 0, stream>>>(dCt, ldct, rows, cols, shift,
                                             dExpA, dExpB, alpha, beta, dC, ldc);
    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}
