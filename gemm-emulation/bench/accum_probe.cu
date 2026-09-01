/* Isolates *accumulation* error from *decomposition* error.
 *
 * The main sweep shows cublasSgemm's forward error growing like sqrt(k) while
 * the 6-MAC emulation's grows roughly like k. Since 6 and 9 MACs give
 * identical error, the dropped cross products cannot be the cause. The other
 * candidate is that cuBLAS's BF16 tensor-core kernel accumulates its k-loop
 * less carefully than its FP32 SIMT kernel does.
 *
 * This probe removes the decomposition from the picture entirely. Inputs are
 * first rounded to BF16, so A0 and B0 represent them *exactly*; the true
 * product is then computable in FP64 with no representation error at all.
 * Running the same exact inputs through
 *
 *   (a) one BF16 tensor-core GEMM with FP32 accumulation, and
 *   (b) cublasSgemm on FP32 copies of the identical values
 *
 * makes any difference in the reported error attributable purely to how the
 * two kernels accumulate.
 */

#include "mpemu/mpemu.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) {                    \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s=(x); if (s!=CUBLAS_STATUS_SUCCESS) {       \
    std::fprintf(stderr,"%s:%d cublas %d\n",__FILE__,__LINE__,(int)s);         \
    std::exit(1);} } while(0)

__device__ __forceinline__ unsigned int splitmix(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (unsigned int)((x ^ (x >> 31)) >> 32);
}

__global__ void fill(float* a, long long n, unsigned long long seed)
{
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    long long st = (long long)gridDim.x*blockDim.x;
    for (; i < n; i += st) {
        float u = (float)(splitmix(seed ^ (unsigned long long)i) >> 8)*5.9604645e-8f;
        a[i] = 2.0f*u - 1.0f;
    }
}

/* Expand a BF16 plane into exact FP32 and FP64 copies. */
__global__ void expand(const __nv_bfloat16* __restrict__ s, long long lds,
                       float* __restrict__ f, double* __restrict__ d,
                       long long ld, long long rows, long long cols)
{
    long long total = rows*cols;
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    long long st = (long long)gridDim.x*blockDim.x;
    for (; i < total; i += st) {
        long long r = i % rows, c = i / rows;
        float v = __bfloat162float(s[c*lds + r]);
        f[c*ld + r] = v;
        d[c*ld + r] = (double)v;
    }
}

__device__ void atomicMaxD(double* addr, double val)
{
    unsigned long long* a = (unsigned long long*)addr; unsigned long long o=*a,as;
    do { as=o; if (__longlong_as_double(as) >= val) break;
         o = atomicCAS(a, as, __double_as_longlong(val)); } while (as!=o);
}

__global__ void errK(const float* C, const double* D, long long ld,
                     long long rows, long long cols, double* out)
{
    __shared__ double sd[256], sr[256];
    double ad=0, ar=0, md=0, mr=0;
    long long total=rows*cols;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for (; i<total; i+=st) {
        long long r=i%rows, c=i/rows;
        double ref=D[c*ld+r], d=(double)C[c*ld+r]-ref;
        ad+=d*d; ar+=ref*ref;
        if (fabs(d)>md) md=fabs(d);
        if (fabs(ref)>mr) mr=fabs(ref);
    }
    int t=threadIdx.x; sd[t]=ad; sr[t]=ar; __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (t<s) { sd[t]+=sd[t+s]; sr[t]+=sr[t+s]; } __syncthreads();
    }
    if (t==0) { atomicAdd(&out[0],sd[0]); atomicAdd(&out[1],sr[0]); }
    atomicMaxD(&out[2],md); atomicMaxD(&out[3],mr);
}

static double relFrob(const float* C, const double* D, long long ld,
                      long long rows, long long cols, double* dOut)
{
    CK(cudaMemset(dOut,0,4*sizeof(double)));
    errK<<<1024,256>>>(C,D,ld,rows,cols,dOut);
    double h[4]; CK(cudaMemcpy(h,dOut,4*sizeof(double),cudaMemcpyDeviceToHost));
    return h[1]>0 ? sqrt(h[0]/h[1]) : 0.0;
}

int main(int argc, char** argv)
{
    const long long m = (argc>1)?atoll(argv[1]):1024;
    const long long n = m;
    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    const double EPS = 5.9604645e-8;  /* FP32 unit roundoff, 2^-24 */
    std::printf("# inputs pre-rounded to BF16, so A0/B0 are EXACT; reference is\n"
                "# cublasDgemm on those exact values. m=n=%lld\n", m);
    std::printf("k,bf16_tc_relfrob,sgemm_relfrob,bf16_tc_over_eps,sgemm_over_eps,ratio\n");

    double* dOut; CK(cudaMalloc(&dOut,4*sizeof(double)));

    for (long long k : {512LL,1024LL,2048LL,4096LL,8192LL,16384LL}) {
        const long long ldsA=mpemuSplitLd(m), strA=mpemuSplitStride(ldsA,k);
        const long long ldsB=mpemuSplitLd(k), strB=mpemuSplitStride(ldsB,n);

        float *dA,*dB,*dAf,*dBf,*dCtc,*dCf;
        double *dAd,*dBd,*dD;
        __nv_bfloat16 *dSA,*dSB;
        CK(cudaMalloc(&dA,(size_t)m*k*4));   CK(cudaMalloc(&dB,(size_t)k*n*4));
        CK(cudaMalloc(&dAf,(size_t)m*k*4));  CK(cudaMalloc(&dBf,(size_t)k*n*4));
        CK(cudaMalloc(&dCtc,(size_t)m*n*4)); CK(cudaMalloc(&dCf,(size_t)m*n*4));
        CK(cudaMalloc(&dAd,(size_t)m*k*8));  CK(cudaMalloc(&dBd,(size_t)k*n*8));
        CK(cudaMalloc(&dD,(size_t)m*n*8));
        CK(cudaMalloc(&dSA,mpemuSplitBytes(m,k,1)));
        CK(cudaMalloc(&dSB,mpemuSplitBytes(k,n,1)));

        fill<<<1024,256>>>(dA,m*k,11ull);
        fill<<<1024,256>>>(dB,k*n,22ull);

        /* 1 split == plain BF16 rounding; A0/B0 then represent dAf/dBf exactly. */
        mpemuSplitBF16(dA,m,m,k,1,dSA,ldsA,strA,0);
        mpemuSplitBF16(dB,k,k,n,1,dSB,ldsB,strB,0);
        expand<<<1024,256>>>(dSA,ldsA,dAf,dAd,m,m,k);
        expand<<<1024,256>>>(dSB,ldsB,dBf,dBd,k,k,n);

        const double one=1.0, zero=0.0;
        CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,
                       &one,dAd,(int)m,dBd,(int)k,&zero,dD,(int)m));

        mpemuGemmBF16(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,
                      dSA,ldsA,dSB,ldsB,0.0f,dCtc,m);

        const float f1=1.0f,f0=0.0f;
        CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,
                       &f1,dAf,(int)m,dBf,(int)k,&f0,dCf,(int)m));
        CK(cudaDeviceSynchronize());

        const double etc = relFrob(dCtc,dD,m,m,n,dOut);
        const double ef  = relFrob(dCf, dD,m,m,n,dOut);
        std::printf("%lld,%.4e,%.4e,%.1f,%.1f,%.2f\n",
                    k, etc, ef, etc/EPS, ef/EPS, (ef>0)?etc/ef:0.0);
        std::fflush(stdout);

        cudaFree(dA);cudaFree(dB);cudaFree(dAf);cudaFree(dBf);
        cudaFree(dCtc);cudaFree(dCf);cudaFree(dAd);cudaFree(dBd);cudaFree(dD);
        cudaFree(dSA);cudaFree(dSB);
    }
    cudaFree(dOut); CB(cublasDestroy(h));
    return 0;
}
