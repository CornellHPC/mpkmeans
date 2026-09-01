/* Correctness checks for routine 1 and the term schedule.
 *
 * The central claim of the 3-way split is that it is *exact*: for FP32 inputs
 * in a sane exponent range, a == b0 + b1 + b2 bit-for-bit. This test verifies
 * that on the GPU, plus the non-vectorised path and the unaligned/tail cases.
 */

#include "mpemu/mpemu.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) {                    \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1);} } while(0)

__device__ __forceinline__ unsigned int splitmix(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (unsigned int)((x ^ (x >> 31)) >> 32);
}

__global__ void fill(float* a, long long n, unsigned long long seed, float expRange)
{
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    long long st = (long long)gridDim.x*blockDim.x;
    for (; i < n; i += st) {
        float u = (float)(splitmix(seed ^ (unsigned long long)i) >> 8) * 5.9604645e-8f;
        float v = 2.0f*u - 1.0f;
        if (expRange > 0.0f) {
            float w = (float)(splitmix(0xABCDull ^ seed ^ (unsigned long long)(i*2654435761ull)) >> 8)
                    * 5.9604645e-8f;
            v *= exp2f((2.0f*w - 1.0f)*expRange);
        }
        a[i] = v;
    }
}

/* nbad  = # entries where the split does not reconstruct bit-exactly
 * maxrel = max |sum b - a| / |a| over those entries */
__global__ void checkKernel(const float* __restrict__ A, long long lda,
                            const __nv_bfloat16* __restrict__ S,
                            long long lds, long long stride, int nsplits,
                            long long rows, long long cols,
                            unsigned long long* nbad, float* maxrel)
{
    long long total = rows*cols;
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    long long st = (long long)gridDim.x*blockDim.x;
    for (; i < total; i += st) {
        long long r = i % rows, c = i / rows;
        float a = A[c*lda + r];
        float s = 0.0f;
        for (int k = 0; k < nsplits; ++k)
            s += __bfloat162float(S[k*stride + c*lds + r]);
        if (s != a) {
            atomicAdd(nbad, 1ull);
            float rel = (a != 0.0f) ? fabsf((s - a)/a) : fabsf(s - a);
            atomicMax((unsigned int*)maxrel, __float_as_uint(rel));
        }
    }
}

static int run(long long rows, long long cols, long long lda, int nsplits,
               float expRange, const char* what)
{
    const long long lds = mpemuSplitLd(rows);
    const long long str = mpemuSplitStride(lds, cols);

    float* dA; __nv_bfloat16* dS;
    unsigned long long* dN; float* dM;
    CK(cudaMalloc(&dA, (size_t)lda*cols*sizeof(float)));
    CK(cudaMalloc(&dS, mpemuSplitBytes(rows, cols, nsplits)));
    CK(cudaMalloc(&dN, sizeof(unsigned long long)));
    CK(cudaMalloc(&dM, sizeof(float)));
    CK(cudaMemset(dN, 0, sizeof(unsigned long long)));
    CK(cudaMemset(dM, 0, sizeof(float)));

    fill<<<256,256>>>(dA, lda*cols, 99ull, expRange);
    mpemuStatus_t st = mpemuSplitBF16(dA, lda, rows, cols, nsplits, dS, lds, str, 0);
    if (st != MPEMU_STATUS_SUCCESS) {
        std::printf("FAIL %-28s split returned %s\n", what, mpemuStatusString(st));
        return 1;
    }
    checkKernel<<<256,256>>>(dA, lda, dS, lds, str, nsplits, rows, cols, dN, dM);
    CK(cudaDeviceSynchronize());

    unsigned long long nbad; float maxrel;
    CK(cudaMemcpy(&nbad, dN, sizeof(nbad), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&maxrel, dM, sizeof(maxrel), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dS); cudaFree(dN); cudaFree(dM);

    const long long total = rows*cols;
    /* nsplits<3 is lossy by construction; only nsplits>=3 must be exact. */
    const bool mustBeExact = (nsplits >= 3);
    const bool ok = mustBeExact ? (nbad == 0) : true;
    std::printf("%s %-28s rows=%-6lld cols=%-6lld lda=%-6lld ns=%d  "
                "inexact=%llu/%lld maxrel=%.3e\n",
                ok ? "ok  " : "FAIL", what, rows, cols, lda, nsplits,
                (unsigned long long)nbad, total, maxrel);
    return ok ? 0 : 1;
}

int main()
{
    int bad = 0;

    /* Exactness of the 3-way split, vectorised path. */
    bad += run(1024, 1024, 1024, 3, 0.0f, "3-split exact, aligned");
    bad += run(4096,  512, 4096, 3, 0.0f, "3-split exact, aligned");

    /* Row-count tail (rows % 4 != 0) and unaligned leading dimension, which
     * force the scalar paths. */
    bad += run(1023, 777, 1023, 3, 0.0f, "3-split, row tail");
    bad += run(1000, 333, 1001, 3, 0.0f, "3-split, odd lda");
    bad += run(   1,   1,    1, 3, 0.0f, "3-split, 1x1");
    bad += run(   7,   3,    9, 3, 0.0f, "3-split, tiny + pad");

    /* Wide exponent range: exact until the 2nd/3rd component underflows BF16.
     * +/-30 stays well inside the paper's recommended [-110,127] window. */
    bad += run(4096, 4096, 4096, 3, 30.0f, "3-split, exp range +/-30");

    /* Fewer splits are lossy but must not error. */
    bad += run(2048, 2048, 2048, 1, 0.0f, "1-split (lossy, ok)");
    bad += run(2048, 2048, 2048, 2, 0.0f, "2-split (lossy, ok)");
    bad += run(2048, 2048, 2048, 4, 0.0f, "4-split exact");

    /* Term schedule: ascending bin order, 1/3/6/9 prefixes for nsplits=3. */
    mpemuTerm_t t[16];
    int nt = mpemuTermSchedule(3, 16, t);
    const int want[9][2] = {{0,0},{0,1},{1,0},{0,2},{1,1},{2,0},{1,2},{2,1},{2,2}};
    bool sched = (nt == 9);
    for (int i = 0; i < nt && i < 9; ++i)
        if (t[i].i != want[i][0] || t[i].j != want[i][1]) sched = false;
    std::printf("%s term schedule nsplits=3 -> %d terms\n", sched ? "ok  " : "FAIL", nt);
    if (!sched) ++bad;

    for (int cap : {1, 3, 6, 9}) {
        int c = mpemuTermSchedule(3, cap, t);
        if (c != cap) { std::printf("FAIL schedule cap %d -> %d\n", cap, c); ++bad; }
    }

    std::printf("\n%s (%d failures)\n", bad ? "FAILED" : "ALL PASSED", bad);
    return bad ? 1 : 0;
}
