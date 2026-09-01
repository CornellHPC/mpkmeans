/* Benchmark and accuracy driver for the multiword fp16 scheme
 * (Fasi, Higham, Lopez, Mary & Mikaitis, MIMS EPrint 2022.3).
 *
 * Compares against BOTH cublasSgemm and cublasDgemm, in runtime and in
 * forward error.
 *
 * Measuring dgemm's own error needs a reference better than fp64. Because the
 * inputs are FP32, every product a*b is EXACT in fp64 (24+24 = 48 <= 53 bits),
 * so the only error in an fp64 evaluation is the summation. Accumulating those
 * exact products with Neumaier compensated summation yields a reference held
 * as an unevaluated pair (hi, lo) worth ~106 bits, which puts dgemm, sgemm and
 * the fp16 multiword schemes on one honest scale.
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

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess) {          \
    std::fprintf(stderr,"%s:%d CUDA: %s\n",__FILE__,__LINE__,                  \
        cudaGetErrorString(e_)); std::exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_=(x); if (s_!=CUBLAS_STATUS_SUCCESS){ \
    std::fprintf(stderr,"%s:%d cuBLAS %d\n",__FILE__,__LINE__,(int)s_);        \
    std::exit(1);} } while(0)
#define MPEMU_CHECK(x) do { mpemuStatus_t s_=(x); if (s_!=MPEMU_STATUS_SUCCESS){\
    std::fprintf(stderr,"%s:%d mpemu: %s\n",__FILE__,__LINE__,                 \
        mpemuStatusString(s_)); std::exit(1);} } while(0)

/* ------------------------------------------------------------- data ----- */

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


/* Uniform on (-0.5, 0.5], the distribution used in the paper's experiments,
 * optionally widened by a random power of two. */
__global__ void fillKernel(float* a, long long n, unsigned long long seed,
                           float expRange)
{
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    long long st = (long long)gridDim.x*blockDim.x;
    for (; i < n; i += st) {
        float v = uniformFloat(seed, i, -0.5f, 0.5f);
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
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for (; i<n; i+=st) d[i]=(double)s[i];
}

/* ------------------------------------------- compensated fp64 reference -- */

#define TS 16

__global__ void refKernel(const float* __restrict__ A, long long lda,
                          const float* __restrict__ B, long long ldb,
                          double* __restrict__ Chi, double* __restrict__ Clo,
                          long long ldc, long long m, long long n, long long k)
{
    __shared__ float As[TS][TS+1], Bs[TS][TS+1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const long long row = (long long)blockIdx.x*TS + tx;
    const long long col = (long long)blockIdx.y*TS + ty;

    double s = 0.0, c = 0.0;
    for (long long t = 0; t < k; t += TS) {
        As[ty][tx] = (row < m && t+ty < k) ? A[(t+ty)*lda + row] : 0.0f;
        Bs[tx][ty] = (col < n && t+tx < k) ? B[col*ldb + (t+tx)] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TS; ++i) {
            /* exact: fp32 x fp32 -> fp64 */
            const double v = (double)As[i][tx] * (double)Bs[i][ty];
            const double tt = s + v;
            if (fabs(s) >= fabs(v)) c += (s - tt) + v;
            else                    c += (v - tt) + s;
            s = tt;
        }
        __syncthreads();
    }
    if (row < m && col < n) { Chi[col*ldc+row] = s; Clo[col*ldc+row] = c; }
}

/* ------------------------------------------------------------ error ----- */

__device__ void atomicMaxD(double* addr, double val)
{
    unsigned long long* a=(unsigned long long*)addr; unsigned long long o=*a,as;
    do { as=o; if (__longlong_as_double(as) >= val) break;
         o = atomicCAS(a, as, __double_as_longlong(val)); } while (as!=o);
}

/* Works for a float or double C against the (hi,lo) reference. */
template <typename T>
__global__ void errKernel(const T* __restrict__ C, long long ldc,
                          const double* __restrict__ Chi,
                          const double* __restrict__ Clo, long long ldr,
                          long long rows, long long cols, double* out)
{
    __shared__ double sd[256], sr[256];
    double ad=0, ar=0, md=0, mr=0;
    const long long total = rows*cols;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long st=(long long)gridDim.x*blockDim.x;
    for (; i<total; i+=st) {
        const long long r=i%rows, c=i/rows;
        const double hi=Chi[c*ldr+r], lo=Clo[c*ldr+r];
        const double ref=hi+lo;
        /* (C - hi) is exact-ish since C ~ hi; then remove lo. */
        const double d=((double)C[c*ldc+r] - hi) - lo;
        ad+=d*d; ar+=ref*ref;
        if (fabs(d)>md) md=fabs(d);
        if (fabs(ref)>mr) mr=fabs(ref);
    }
    const int t=threadIdx.x; sd[t]=ad; sr[t]=ar; __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1) {
        if (t<s){ sd[t]+=sd[t+s]; sr[t]+=sr[t+s]; } __syncthreads();
    }
    if (t==0){ atomicAdd(&out[0],sd[0]); atomicAdd(&out[1],sr[0]); }
    atomicMaxD(&out[2],md); atomicMaxD(&out[3],mr);
}

struct Err { double frob, maxrel; };

template <typename T>
static Err measure(const T* C, long long ldc, const double* Chi,
                   const double* Clo, long long ldr,
                   long long rows, long long cols, double* dOut)
{
    CUDA_CHECK(cudaMemset(dOut,0,4*sizeof(double)));
    errKernel<T><<<1024,256>>>(C,ldc,Chi,Clo,ldr,rows,cols,dOut);
    CUDA_CHECK(cudaGetLastError());
    double h[4]; CUDA_CHECK(cudaMemcpy(h,dOut,4*sizeof(double),cudaMemcpyDeviceToHost));
    Err e; e.frob=(h[1]>0)?sqrt(h[0]/h[1]):0.0; e.maxrel=(h[3]>0)?h[2]/h[3]:0.0;
    return e;
}

/* ------------------------------------------------------------- timer ---- */

struct Timer {
    cudaEvent_t a,b;
    Timer(){ cudaEventCreate(&a); cudaEventCreate(&b);} 
    ~Timer(){ cudaEventDestroy(a); cudaEventDestroy(b);} 
    void start(){ CUDA_CHECK(cudaEventRecord(a)); }
    float stop(){ CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
                  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms; }
};

struct Shape { long long m,n,k; const char* tag; };
struct Config { int p; int macs; const char* name; };

int main(int argc, char** argv)
{
    int reps=5, warmup=2;
    long long maxDim=0;
    double refFlopCap=1.4e11;      /* ~4096^3; the reference costs ~4x dgemm */
    float expRange=0.0f;
    bool squareOnly=false, rectOnly=false, noCheck=false;

    for (int i=1;i<argc;++i){
        std::string a=argv[i];
        auto val=[&]{ return (i+1<argc)?argv[++i]:nullptr; };
        if      (a=="--reps")       reps=atoi(val());
        else if (a=="--warmup")     warmup=atoi(val());
        else if (a=="--max-dim")    maxDim=atoll(val());
        else if (a=="--ref-flops")  refFlopCap=atof(val());
        else if (a=="--exp-range")  expRange=(float)atof(val());
        else if (a=="--square")     squareOnly=true;
        else if (a=="--rect")       rectOnly=true;
        else if (a=="--no-check")   noCheck=true;
        else if (a=="--help"){
            std::printf("usage: %s [--reps N] [--warmup N] [--max-dim N]\n"
                        "       [--ref-flops F] [--exp-range F] [--no-check]\n"
                        "       [--square|--rect]\n", argv[0]); return 0; }
    }

    std::vector<Shape> square={
        {1024,1024,1024,"square"},{2048,2048,2048,"square"},
        {3072,3072,3072,"square"},{4096,4096,4096,"square"},
        {6144,6144,6144,"square"},{8192,8192,8192,"square"},
        {12288,12288,12288,"square"},{16384,16384,16384,"square"}};
    std::vector<Shape> rect={
        {16384,16384,512,"k-thin"},{16384,16384,2048,"k-thin"},
        {16384,2048,16384,"n-thin"},{2048,16384,16384,"m-thin"},
        {4096,4096,16384,"k-fat"}};
    std::vector<Shape> shapes;
    if(!rectOnly)   shapes.insert(shapes.end(),square.begin(),square.end());
    if(!squareOnly) shapes.insert(shapes.end(),rect.begin(),rect.end());
    if(maxDim>0){ std::vector<Shape> keep;
        for(const Shape&s:shapes) if(s.m<=maxDim&&s.n<=maxDim&&s.k<=maxDim) keep.push_back(s);
        shapes.swap(keep); }

    /* p=2 / 3 MACs is the paper's recommended "double-fp16". The p=2/4-MAC
     * row exists to show the i+j>p+1 term really is negligible. */
    const std::vector<Config> configs={
        {1,1,"fp16"},
        {2,3,"double-fp16"},
        {2,4,"double-fp16-all4"},
        {3,6,"triple-fp16"}};

    int dev=0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    std::fprintf(stderr,"# device: %s sm_%d%d %.1f GB\n# reps=%d exp-range=%.1f\n",
                 prop.name,prop.major,prop.minor,prop.totalGlobalMem/1073741824.0,
                 reps,expRange);

    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    CUBLAS_CHECK(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));

    std::printf("shape,m,n,k,method,p,macs,scaled,ms_gemm,ms_total,tflops_gemm,"
                "vs_sgemm,vs_dgemm,rel_frob,rel_max,subnormal_frac\n");

    double* dErr; CUDA_CHECK(cudaMalloc(&dErr,4*sizeof(double)));
    unsigned long long* dCounts; CUDA_CHECK(cudaMalloc(&dCounts,3*sizeof(unsigned long long)));

    for (const Shape& sh : shapes) {
        const long long m=sh.m,n=sh.n,k=sh.k;
        const double flops=2.0*(double)m*(double)n*(double)k;
        const long long ldA=m,ldB=k,ldC=m;
        const long long ldsA=mpemuSplitLd(m), strA=mpemuSplitStride(ldsA,k);
        const long long ldsB=mpemuSplitLd(k), strB=mpemuSplitStride(ldsB,n);

        float *dA,*dB,*dCs,*dCe; double *dAd,*dBd,*dCd;
        __half *dSA,*dSB;
        CUDA_CHECK(cudaMalloc(&dA,(size_t)m*k*4));  CUDA_CHECK(cudaMalloc(&dB,(size_t)k*n*4));
        CUDA_CHECK(cudaMalloc(&dCs,(size_t)m*n*4)); CUDA_CHECK(cudaMalloc(&dCe,(size_t)m*n*4));
        CUDA_CHECK(cudaMalloc(&dAd,(size_t)m*k*8)); CUDA_CHECK(cudaMalloc(&dBd,(size_t)k*n*8));
        CUDA_CHECK(cudaMalloc(&dCd,(size_t)m*n*8));
        CUDA_CHECK(cudaMalloc(&dSA,mpemuSplitBytes(m,k,MPEMU_MAX_SPLITS)));
        CUDA_CHECK(cudaMalloc(&dSB,mpemuSplitBytes(k,n,MPEMU_MAX_SPLITS)));

        fillKernel<<<1024,256>>>(dA,m*k,1234ull,expRange);
        fillKernel<<<1024,256>>>(dB,k*n,5678ull,expRange);
        f2dKernel<<<1024,256>>>(dA,dAd,m*k);
        f2dKernel<<<1024,256>>>(dB,dBd,k*n);
        CUDA_CHECK(cudaGetLastError());

        /* -------- compensated reference (never timed) -------- */
        const bool doCheck = !noCheck && flops <= refFlopCap;
        double *dRhi=nullptr,*dRlo=nullptr;
        if (doCheck) {
            CUDA_CHECK(cudaMalloc(&dRhi,(size_t)m*n*8));
            CUDA_CHECK(cudaMalloc(&dRlo,(size_t)m*n*8));
            dim3 blk(TS,TS), grd((unsigned)((m+TS-1)/TS),(unsigned)((n+TS-1)/TS));
            refKernel<<<grd,blk>>>(dA,ldA,dB,ldB,dRhi,dRlo,ldC,m,n,k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        char eb[80];
        auto fmtErr=[&](const Err& e){
            if (doCheck) std::snprintf(eb,sizeof eb,"%.3e,%.3e",e.frob,e.maxrel);
            else         std::snprintf(eb,sizeof eb,",");
            return eb; };

        const float f1=1.0f,f0=0.0f; const double d1=1.0,d0=0.0;
        Timer t;

        /* ------------------------- cublasDgemm ------------------------- */
        for(int i=0;i<warmup;++i)
            CUBLAS_CHECK(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,
                &d1,dAd,(int)ldA,dBd,(int)ldB,&d0,dCd,(int)ldC));
        CUDA_CHECK(cudaDeviceSynchronize());
        t.start();
        for(int i=0;i<reps;++i)
            CUBLAS_CHECK(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,
                &d1,dAd,(int)ldA,dBd,(int)ldB,&d0,dCd,(int)ldC));
        const double dgemmMs=t.stop()/reps;
        Err dErrS{0,0}; if(doCheck) dErrS=measure<double>(dCd,ldC,dRhi,dRlo,ldC,m,n,dErr);

        /* ------------------------- cublasSgemm ------------------------- */
        for(int i=0;i<warmup;++i)
            CUBLAS_CHECK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,
                &f1,dA,(int)ldA,dB,(int)ldB,&f0,dCs,(int)ldC));
        CUDA_CHECK(cudaDeviceSynchronize());
        t.start();
        for(int i=0;i<reps;++i)
            CUBLAS_CHECK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,
                &f1,dA,(int)ldA,dB,(int)ldB,&f0,dCs,(int)ldC));
        const double sgemmMs=t.stop()/reps;
        Err sErrS{0,0}; if(doCheck) sErrS=measure<float>(dCs,ldC,dRhi,dRlo,ldC,m,n,dErr);

        std::printf("%s,%lld,%lld,%lld,cublasDgemm,,,,%.4f,%.4f,%.2f,%.3f,1.000,%s,\n",
            sh.tag,m,n,k,dgemmMs,dgemmMs,flops/(dgemmMs*1e9),sgemmMs/dgemmMs,fmtErr(dErrS));
        std::printf("%s,%lld,%lld,%lld,cublasSgemm,,,,%.4f,%.4f,%.2f,1.000,%.3f,%s,\n",
            sh.tag,m,n,k,sgemmMs,sgemmMs,flops/(sgemmMs*1e9),dgemmMs/sgemmMs,fmtErr(sErrS));
        std::fflush(stdout);

        /* ------------------------- multiword fp16 ---------------------- */
        for (int scaled=0; scaled<2; ++scaled) {
            float sA=1.0f, sB=1.0f;
            if (scaled) {
                MPEMU_CHECK(mpemuAutoScaleFP16(dA,ldA,m,k,&sA,0));
                MPEMU_CHECK(mpemuAutoScaleFP16(dB,ldB,k,n,&sB,0));
            }
            for (const Config& cf : configs) {
                /* split (timed separately) */
                for(int i=0;i<warmup;++i){
                    MPEMU_CHECK(mpemuSplitFP16(dA,ldA,m,k,cf.p,sA,dSA,ldsA,strA,0));
                    MPEMU_CHECK(mpemuSplitFP16(dB,ldB,k,n,cf.p,sB,dSB,ldsB,strB,0)); }
                CUDA_CHECK(cudaDeviceSynchronize());
                t.start();
                for(int i=0;i<reps;++i){
                    MPEMU_CHECK(mpemuSplitFP16(dA,ldA,m,k,cf.p,sA,dSA,ldsA,strA,0));
                    MPEMU_CHECK(mpemuSplitFP16(dB,ldB,k,n,cf.p,sB,dSB,ldsB,strB,0)); }
                const double splitMs=t.stop()/reps;

                /* subnormal / overflow diagnostic on A's split */
                CUDA_CHECK(cudaMemset(dCounts,0,3*sizeof(unsigned long long)));
                MPEMU_CHECK(mpemuCheckSplitFP16(dA,ldA,m,k,cf.p,sA,dSA,ldsA,strA,dCounts,0));
                CUDA_CHECK(cudaDeviceSynchronize());
                unsigned long long hc[3];
                CUDA_CHECK(cudaMemcpy(hc,dCounts,3*sizeof(hc[0]),cudaMemcpyDeviceToHost));
                const double subFrac=(double)hc[1]/(double)(hc[0]+hc[1]+hc[2]);

                for(int i=0;i<warmup;++i)
                    MPEMU_CHECK(mpemuGemmMultiword(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,
                        dSA,ldsA,strA,sA,dSB,ldsB,strB,sB,cf.p,cf.macs,0.0f,dCe,ldC));
                CUDA_CHECK(cudaDeviceSynchronize());
                t.start();
                for(int i=0;i<reps;++i)
                    MPEMU_CHECK(mpemuGemmMultiword(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,
                        dSA,ldsA,strA,sA,dSB,ldsB,strB,sB,cf.p,cf.macs,0.0f,dCe,ldC));
                const double gemmMs=t.stop()/reps;
                const double totalMs=gemmMs+splitMs;

                Err e{0,0}; if(doCheck) e=measure<float>(dCe,ldC,dRhi,dRlo,ldC,m,n,dErr);

                std::printf("%s,%lld,%lld,%lld,%s,%d,%d,%d,%.4f,%.4f,%.2f,%.3f,%.3f,%s,%.4f\n",
                    sh.tag,m,n,k,cf.name,cf.p,cf.macs,scaled,gemmMs,totalMs,
                    flops/(gemmMs*1e9),sgemmMs/gemmMs,dgemmMs/gemmMs,fmtErr(e),subFrac);
                std::fflush(stdout);
            }
            std::fprintf(stderr,"# %s %lldx%lldx%lld scaled=%d scaleA=%g scaleB=%g\n",
                         sh.tag,m,n,k,scaled,sA,sB);
        }

        cudaFree(dA);cudaFree(dB);cudaFree(dCs);cudaFree(dCe);
        cudaFree(dAd);cudaFree(dBd);cudaFree(dCd);cudaFree(dSA);cudaFree(dSB);
        if(doCheck){ cudaFree(dRhi); cudaFree(dRlo); }
    }
    cudaFree(dErr); cudaFree(dCounts);
    CUBLAS_CHECK(cublasDestroy(h));
    return 0;
}
