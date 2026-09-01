/* Benchmark and accuracy driver for mpemu.
 *
 * Compares BF16-emulated FP32 GEMM (nsplits=3, 1/3/6/9 multiply-accumulates)
 * against cublasSgemm on the same shapes. Forward error for every method,
 * cublasSgemm included, is measured against a cublasDgemm reference computed
 * from exact FP64 copies of the same FP32 inputs -- SGEMM is not itself exact,
 * so it cannot serve as its own yardstick.
 *
 * Everything stays resident on the GPU; only scalar statistics come back.
 */

#include "mpemu/mpemu.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ util -- */

#define CUDA_CHECK(x) do {                                                     \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
        std::fprintf(stderr, "%s:%d CUDA: %s\n", __FILE__, __LINE__,           \
                     cudaGetErrorString(e_));                                  \
        std::exit(1);                                                          \
    } } while (0)

#define CUBLAS_CHECK(x) do {                                                   \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
        std::fprintf(stderr, "%s:%d cuBLAS status %d\n", __FILE__, __LINE__,   \
                     (int)s_);                                                 \
        std::exit(1);                                                          \
    } } while (0)

#define MPEMU_CHECK(x) do {                                                    \
    mpemuStatus_t s_ = (x);                                                    \
    if (s_ != MPEMU_STATUS_SUCCESS) {                                          \
        std::fprintf(stderr, "%s:%d mpemu: %s\n", __FILE__, __LINE__,          \
                     mpemuStatusString(s_));                                   \
        std::exit(1);                                                          \
    } } while (0)

/* ------------------------------------------------------------- fill data -- */

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* A properly-rounded FP32 sample of a real uniform variate.
 *
 * Drawing u on a 2^-24 grid (e.g. (hash>>8)*2^-24) would give values with at
 * most 23 significant bits, which a 2-word fp16 split can represent EXACTLY --
 * flattering every scheme measured here. Building the variate in double first
 * and rounding once to float yields a generic full 24-bit significand. */
__device__ __forceinline__ float uniformFloat(unsigned long long seed,
                                              long long i, float lo, float hi)
{
    const unsigned long long h = splitmix64(seed ^ (unsigned long long)i);
    const double u = (double)(h >> 11) * (1.0 / 9007199254740992.0);  /* [0,1) */
    return (float)((double)lo + u * ((double)hi - (double)lo));
}


/* Uniform in [-1,1], optionally scaled by 2^u with u uniform in
 * [-expRange, expRange] to exercise the wide-dynamic-range regime. */
__global__ void fillKernel(float* a, long long n, unsigned long long seed,
                           float expRange)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float v = uniformFloat(seed, i, -1.0f, 1.0f);
        if (expRange > 0.0f) {
            const float w = uniformFloat(0x5DEECE66Dull ^ seed, i, -1.0f, 1.0f);
            v *= exp2f(w * expRange);
        }
        a[i] = v;
    }
}

__global__ void f2dKernel(const float* __restrict__ s, double* __restrict__ d,
                          long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) d[i] = (double)s[i];
}

/* -------------------------------------------------------- error reduction -- */

__device__ void atomicMaxDouble(double* addr, double val)
{
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(a, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

/* out[0] = sum (C-D)^2, out[1] = sum D^2, out[2] = max|C-D|, out[3] = max|D| */
__global__ void errKernel(const float* __restrict__ C, long long ldc,
                          const double* __restrict__ D, long long ldd,
                          long long rows, long long cols, double* out)
{
    __shared__ double sDiff[256], sRef[256];
    double aDiff = 0.0, aRef = 0.0, mDiff = 0.0, mRef = 0.0;

    long long total = rows * cols;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < total; i += stride) {
        long long r = i % rows, c = i / rows;
        double ref = D[c * ldd + r];
        double d = (double)C[c * ldc + r] - ref;
        double ad = fabs(d), ar = fabs(ref);
        aDiff += d * d;
        aRef  += ref * ref;
        if (ad > mDiff) mDiff = ad;
        if (ar > mRef)  mRef  = ar;
    }

    int t = threadIdx.x;
    sDiff[t] = aDiff; sRef[t] = aRef;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) { sDiff[t] += sDiff[t + s]; sRef[t] += sRef[t + s]; }
        __syncthreads();
    }
    if (t == 0) { atomicAdd(&out[0], sDiff[0]); atomicAdd(&out[1], sRef[0]); }
    atomicMaxDouble(&out[2], mDiff);
    atomicMaxDouble(&out[3], mRef);
}

struct ErrStats { double frob; double maxrel; };

static ErrStats measure(const float* C, long long ldc,
                        const double* D, long long ldd,
                        long long rows, long long cols, double* dOut)
{
    CUDA_CHECK(cudaMemset(dOut, 0, 4 * sizeof(double)));
    errKernel<<<1024, 256>>>(C, ldc, D, ldd, rows, cols, dOut);
    CUDA_CHECK(cudaGetLastError());
    double h[4];
    CUDA_CHECK(cudaMemcpy(h, dOut, 4 * sizeof(double), cudaMemcpyDeviceToHost));
    ErrStats s;
    s.frob   = (h[1] > 0.0) ? sqrt(h[0] / h[1]) : 0.0;
    s.maxrel = (h[3] > 0.0) ? h[2] / h[3] : 0.0;
    return s;
}

/* ------------------------------------------------------------------ time -- */

struct Timer {
    cudaEvent_t a, b;
    Timer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { CUDA_CHECK(cudaEventRecord(a)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

/* --------------------------------------------------------------- shapes --- */

struct Shape { long long m, n, k; const char* tag; };

int main(int argc, char** argv)
{
    int    reps       = 5;
    int    warmup     = 2;
    int    nsplits    = 3;
    long long maxDim   = 0;      /* 0 = no limit; smoke-test filter */
    double refBytesCap = 8.0e9;  /* FP64 reference working-set cap */
    double refFlopCap  = 4.0e12; /* FP64 reference cost cap */
    bool   noCheck     = false;
    float  expRange   = 0.0f;
    bool   squareOnly = false;
    bool   rectOnly   = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if      (a == "--reps")      reps     = atoi(val());
        else if (a == "--warmup")    warmup   = atoi(val());
        else if (a == "--splits")    nsplits  = atoi(val());
        else if (a == "--max-dim")   maxDim   = atoll(val());
        else if (a == "--ref-bytes") refBytesCap = atof(val());
        else if (a == "--ref-flops") refFlopCap  = atof(val());
        else if (a == "--no-check")  noCheck  = true;
        else if (a == "--exp-range") expRange = (float)atof(val());
        else if (a == "--square")    squareOnly = true;
        else if (a == "--rect")      rectOnly   = true;
        else if (a == "--help") {
            std::printf("usage: %s [--reps N] [--warmup N] [--splits N]\n"
                        "       [--max-dim N] [--no-check] [--ref-bytes B] [--ref-flops F]\n"
                        "       [--exp-range F] [--square|--rect]\n", argv[0]);
            return 0;
        }
    }

    std::vector<Shape> square = {
        {1024,1024,1024,"square"},   {2048,2048,2048,"square"},
        {3072,3072,3072,"square"},   {4096,4096,4096,"square"},
        {6144,6144,6144,"square"},   {8192,8192,8192,"square"},
        {12288,12288,12288,"square"},{16384,16384,16384,"square"},
    };
    std::vector<Shape> rect = {
        {16384,16384,  512,"k-thin"},   {16384,16384, 1024,"k-thin"},
        {16384,16384, 2048,"k-thin"},   {16384,16384, 4096,"k-thin"},
        {16384,  512,16384,"n-thin"},   {16384, 2048,16384,"n-thin"},
        {  512,16384,16384,"m-thin"},   { 2048,16384,16384,"m-thin"},
        { 4096, 4096,16384,"k-fat"},
    };
    std::vector<Shape> shapes;
    if (!rectOnly)   shapes.insert(shapes.end(), square.begin(), square.end());
    if (!squareOnly) shapes.insert(shapes.end(), rect.begin(),   rect.end());
    if (maxDim > 0) {
        std::vector<Shape> keep;
        for (const Shape& s : shapes)
            if (s.m <= maxDim && s.n <= maxDim && s.k <= maxDim) keep.push_back(s);
        shapes.swap(keep);
    }

    /* -------------------------------------------------------- device --- */
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::fprintf(stderr, "# device: %s  sm_%d%d  %.1f GB  %d SMs\n",
                 prop.name, prop.major, prop.minor,
                 prop.totalGlobalMem / 1073741824.0, prop.multiProcessorCount);
    std::fprintf(stderr, "# nsplits=%d reps=%d warmup=%d exp-range=%.1f "
                 "TF32_OVERRIDE=%s\n", nsplits, reps, warmup, expRange,
                 getenv("NVIDIA_TF32_OVERRIDE") ? getenv("NVIDIA_TF32_OVERRIDE") : "(unset)");

    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    /* Keep cublasSgemm on real FP32: no TF32 tensor-core substitution. */
    CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    const int maxTerms = nsplits * nsplits;
    std::vector<int> macList;
    for (int t = 1; t <= maxTerms; ++t)
        if (nsplits != 3 || t == 1 || t == 3 || t == 6 || t == 9) macList.push_back(t);

    std::printf("shape,m,n,k,method,macs,ms_gemm,ms_total,tflops_gemm,tflops_total,"
                "speedup_gemm,speedup_total,rel_frob,rel_max\n");

    double* dErr = nullptr;
    CUDA_CHECK(cudaMalloc(&dErr, 4 * sizeof(double)));

    for (const Shape& sh : shapes) {
        const long long m = sh.m, n = sh.n, k = sh.k;
        const double flops = 2.0 * (double)m * (double)n * (double)k;

        const long long ldA = m, ldB = k, ldC = m;
        const long long ldsA = mpemuSplitLd(m),  strA = mpemuSplitStride(ldsA, k);
        const long long ldsB = mpemuSplitLd(k),  strB = mpemuSplitStride(ldsB, n);

        float *dA=nullptr,*dB=nullptr,*dCs=nullptr,*dCe=nullptr;
        __nv_bfloat16 *dSA=nullptr,*dSB=nullptr;
        CUDA_CHECK(cudaMalloc(&dA,  (size_t)m*k*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB,  (size_t)k*n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCs, (size_t)m*n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCe, (size_t)m*n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dSA, mpemuSplitBytes(m, k, nsplits)));
        CUDA_CHECK(cudaMalloc(&dSB, mpemuSplitBytes(k, n, nsplits)));

        fillKernel<<<1024,256>>>(dA, m*k, 1234ull, expRange);
        fillKernel<<<1024,256>>>(dB, k*n, 5678ull, expRange);
        CUDA_CHECK(cudaGetLastError());

        /* ---- FP64 reference (error yardstick only, never timed) ----
         * Gated on the reference's own cost, not on matrix dimension, so
         * rectangular shapes still get error numbers even when one dimension
         * is large. */
        const double refBytes = 8.0 * ((double)m*n + (double)m*k + (double)k*n);
        const bool doCheck = !noCheck && refBytes <= refBytesCap
                                      && flops <= refFlopCap;
        double *dAd=nullptr,*dBd=nullptr,*dD=nullptr;
        if (doCheck) {
            CUDA_CHECK(cudaMalloc(&dAd, (size_t)m*k*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dBd, (size_t)k*n*sizeof(double)));
            CUDA_CHECK(cudaMalloc(&dD,  (size_t)m*n*sizeof(double)));
            f2dKernel<<<1024,256>>>(dA, dAd, m*k);
            f2dKernel<<<1024,256>>>(dB, dBd, k*n);
            const double one = 1.0, zero = 0.0;
            CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)m,(int)n,(int)k,
                                     &one, dAd, (int)ldA, dBd, (int)ldB,
                                     &zero, dD, (int)ldC));
        }

        const float one = 1.0f, zero = 0.0f;

        /* ------------------------- baseline: cublasSgemm ------------- */
        for (int i = 0; i < warmup; ++i)
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)m,(int)n,(int)k,
                                     &one, dA,(int)ldA, dB,(int)ldB, &zero, dCs,(int)ldC));
        CUDA_CHECK(cudaDeviceSynchronize());
        Timer t;
        t.start();
        for (int i = 0; i < reps; ++i)
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)m,(int)n,(int)k,
                                     &one, dA,(int)ldA, dB,(int)ldB, &zero, dCs,(int)ldC));
        const double sgemmMs = t.stop() / reps;

        ErrStats sgemmErr{0,0};
        if (doCheck) sgemmErr = measure(dCs, ldC, dD, ldC, m, n, dErr);

        char errbuf[64];
        auto fmtErr = [&](const ErrStats& e) {
            if (doCheck) std::snprintf(errbuf, sizeof errbuf, "%.3e,%.3e", e.frob, e.maxrel);
            else         std::snprintf(errbuf, sizeof errbuf, ",");   /* reference skipped */
            return errbuf;
        };

        std::printf("%s,%lld,%lld,%lld,cublasSgemm,0,%.4f,%.4f,%.2f,%.2f,1.000,1.000,%s\n",
                    sh.tag, m,n,k, sgemmMs, sgemmMs,
                    flops/(sgemmMs*1e9), flops/(sgemmMs*1e9), fmtErr(sgemmErr));
        std::fflush(stdout);

        /* ------------------------- split cost (timed separately) ----- */
        for (int i = 0; i < warmup; ++i) {
            MPEMU_CHECK(mpemuSplitBF16(dA, ldA, m, k, nsplits, dSA, ldsA, strA, 0));
            MPEMU_CHECK(mpemuSplitBF16(dB, ldB, k, n, nsplits, dSB, ldsB, strB, 0));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t.start();
        for (int i = 0; i < reps; ++i) {
            MPEMU_CHECK(mpemuSplitBF16(dA, ldA, m, k, nsplits, dSA, ldsA, strA, 0));
            MPEMU_CHECK(mpemuSplitBF16(dB, ldB, k, n, nsplits, dSB, ldsB, strB, 0));
        }
        const double splitMs = t.stop() / reps;

        /* ------------------------- emulated ------------------------- */
        for (int macs : macList) {
            for (int i = 0; i < warmup; ++i)
                MPEMU_CHECK(mpemuGemmEmulated(h, CUBLAS_OP_N, CUBLAS_OP_N, m,n,k, 1.0f,
                                              dSA, ldsA, strA, dSB, ldsB, strB,
                                              nsplits, macs, 0.0f, dCe, ldC));
            CUDA_CHECK(cudaDeviceSynchronize());
            t.start();
            for (int i = 0; i < reps; ++i)
                MPEMU_CHECK(mpemuGemmEmulated(h, CUBLAS_OP_N, CUBLAS_OP_N, m,n,k, 1.0f,
                                              dSA, ldsA, strA, dSB, ldsB, strB,
                                              nsplits, macs, 0.0f, dCe, ldC));
            const double gemmMs  = t.stop() / reps;
            const double totalMs = gemmMs + splitMs;

            ErrStats e{0,0};
            if (doCheck) e = measure(dCe, ldC, dD, ldC, m, n, dErr);

            std::printf("%s,%lld,%lld,%lld,mpemu,%d,%.4f,%.4f,%.2f,%.2f,%.3f,%.3f,%s\n",
                        sh.tag, m,n,k, macs, gemmMs, totalMs,
                        flops/(gemmMs*1e9), flops/(totalMs*1e9),
                        sgemmMs/gemmMs, sgemmMs/totalMs, fmtErr(e));
            std::fflush(stdout);
        }

        std::fprintf(stderr, "# %s %lldx%lldx%lld split=%.3f ms (%.1f%% of sgemm)\n",
                     sh.tag, m,n,k, splitMs, 100.0*splitMs/sgemmMs);

        cudaFree(dA); cudaFree(dB); cudaFree(dCs); cudaFree(dCe);
        cudaFree(dSA); cudaFree(dSB);
        if (doCheck) { cudaFree(dAd); cudaFree(dBd); cudaFree(dD); }
    }

    cudaFree(dErr);
    CUBLAS_CHECK(cublasDestroy(h));
    return 0;
}
