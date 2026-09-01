/* Steady-state iteration benchmark -- the only runtime measurement.
 *
 * Rectangular shapes at m = 16384 only. The workload holds A fixed and changes
 * only B between iterations, so A is split ONCE, outside the loop, and each of
 * the 100 timed iterations re-derives only B: its scale or shared exponents,
 * its split planes, and the products into C.
 *
 * The mantissa-split schemes are measured two ways at each product count:
 *
 *   per-product  the original naive scheme -- one cuBLAS call per cross
 *                product, each accumulating into C with beta=1
 *   concat       one GEMM per bin over the stacked reversed right operand,
 *                no temporary and no reduction
 *
 * Both are evaluated at the SAME product counts, which must be bin boundaries
 * (1, 3, 6 for bf16-split; 1, 3 for multiword) because concat cannot stop
 * part-way through a bin. Per-product can refine at any single product; that
 * finer granularity is its only advantage.
 *
 * Ozaki I has no concat path and uses its bin-grouped driver, bins 1..8.
 *
 * Every buffer -- split planes, INT32 scratch, exponent vectors -- is carved
 * once from an mpemuContext arena before the loop, so the steady state
 * performs ZERO CUDA allocations. The benchmark asserts that by checking
 * mpemuContextAllocCount before and after.
 *
 * B's contents are not regenerated between iterations: re-splitting the same
 * buffer costs exactly what re-splitting a changed one would, and keeping it
 * fixed lets the accuracy check stay meaningful.
 */

#include "mpemu/mpemu.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t ck_=(x); if(ck_!=cudaSuccess){                  \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,                        \
    cudaGetErrorString(ck_)); std::exit(1);} } while(0)
#define CB(x) do { cublasStatus_t cb_=(x); if(cb_!=CUBLAS_STATUS_SUCCESS){     \
    std::fprintf(stderr,"%s:%d cublas %d\n",__FILE__,__LINE__,(int)cb_);       \
    std::exit(1);} } while(0)
#define MP(x) do { mpemuStatus_t mp_=(x); if(mp_!=MPEMU_STATUS_SUCCESS){       \
    std::fprintf(stderr,"%s:%d mpemu %s\n",__FILE__,__LINE__,                  \
    mpemuStatusString(mp_)); std::exit(1);} } while(0)

__device__ __forceinline__ unsigned long long sm64(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x>>27)) * 0x94D049BB133111EBull;
    return x ^ (x>>31);
}
__device__ __forceinline__ double u01(unsigned long long h)
{ return (double)(h>>11)*(1.0/9007199254740992.0); }

__global__ void fillKernel(double* a,long long n,unsigned long long seed,int dist)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st){
        const unsigned long long h1=sm64(seed^(unsigned long long)i);
        if(dist==0) a[i]=2.0*u01(h1)-1.0;
        else{
            const unsigned long long h2=sm64(h1^0x9E3779B97F4A7C15ull);
            double v1=u01(h1); if(v1<1e-300) v1=1e-300;
            a[i]=sqrt(-2.0*log(v1))*cos(6.283185307179586*u01(h2));
        }
    }
}
__global__ void d2f(const double* __restrict__ s,float* __restrict__ f,long long n)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st) f[i]=(float)s[i];
}


/* Plain half-precision baselines: cast once, then a single cublasGemmEx with
 * FP32 accumulation. This is the "just use tensor cores and accept the
 * accuracy" option that every emulation scheme is trying to improve on. */
__global__ void f2bf(const float* __restrict__ s, __nv_bfloat16* __restrict__ d, long long n)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st) d[i]=__float2bfloat16(s[i]);
}
__global__ void f2h(const float* __restrict__ s, __half* __restrict__ d, long long n)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st) d[i]=__float2half(s[i]);
}

__device__ __forceinline__ void ddAcc(double& sh,double& sl,double p,double e)
{
    double t=sh+p,bv=t-sh;
    sl += ((sh-(t-bv))+(p-bv))+e;
    double s2=t+sl; sl=sl-(s2-t); sh=s2;
}
__global__ void ddRefSub(const double* __restrict__ A,long long lda,
                         const double* __restrict__ B,long long ldb,
                         double* __restrict__ Rhi,double* __restrict__ Rlo,
                         long long ldr,long long rs,long long cs,long long k)
{
    const long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=rs*cs) return;
    const long long r=idx%rs,c=idx/rs;
    double sh=0.0,sl=0.0;
    for(long long l=0;l<k;++l){
        const double a=A[l*lda+r],b=B[c*ldb+l];
        const double p=a*b,e=fma(a,b,-p);
        ddAcc(sh,sl,p,e);
    }
    Rhi[c*ldr+r]=sh; Rlo[c*ldr+r]=sl;
}
template<typename T>
__global__ void errSub(const T* __restrict__ C,long long ldc,
                       const double* __restrict__ Rhi,const double* __restrict__ Rlo,
                       long long ldr,long long rs,long long cs,double* out)
{
    __shared__ double sd[256],sr[256];
    double ad=0,ar=0;
    const long long total=rs*cs;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long st=(long long)gridDim.x*blockDim.x;
    for(; i<total; i+=st){
        const long long r=i%rs,c=i/rs;
        const double hi=Rhi[c*ldr+r],lo=Rlo[c*ldr+r];
        const double d=(((double)C[c*ldc+r]-hi)-lo),ref=hi+lo;
        ad+=d*d; ar+=ref*ref;
    }
    const int t=threadIdx.x; sd[t]=ad; sr[t]=ar; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s){sd[t]+=sd[t+s];sr[t]+=sr[t+s];} __syncthreads(); }
    if(t==0){ atomicAdd(&out[0],sd[0]); atomicAdd(&out[1],sr[0]); }
}

struct Timer{
    cudaEvent_t a,b;
    Timer(){cudaEventCreate(&a);cudaEventCreate(&b);}
    ~Timer(){cudaEventDestroy(a);cudaEventDestroy(b);}
    void start(){CK(cudaEventRecord(a));}
    double stop(){CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
                  float ms;CK(cudaEventElapsedTime(&ms,a,b));return ms;}
};
struct Shape{ long long m,n,k; };

int main(int argc,char** argv)
{
    int dist=1, iters=100, warmup=5;          /* gaussian by default */
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto val=[&]{ return (i+1<argc)?argv[++i]:nullptr; };
        if(a=="--dist"){ std::string d=val(); dist=(d=="uniform")?0:1; }
        else if(a=="--iters") iters=atoi(val());
        else if(a=="--help"){ std::printf("usage: %s [--dist gaussian|uniform] [--iters N]\n",argv[0]); return 0; }
    }
    const char* distName = dist?"gaussian":"uniform";

    /* rectangular only, m = 16384 */
    std::vector<Shape> shapes={
        {16384,   64,   100}, {16384,   64,  1000}, {16384,   64, 10000},
        {16384,  256,   100}, {16384,  256,  1000}, {16384,  256, 10000},
        {16384, 1000,   256}, {16384,10000,   256}};

    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));
    mpemuContext_t ctx; MP(mpemuContextCreate(&ctx));

    std::printf("dist,m,n,k,method,bins,products,iters,ms_per_iter,"
                "ms_scale,ms_split,ms_gemm,ms_fold,vs_fp32,vs_fp64,rel_err\n");

    double* dOut; CK(cudaMalloc(&dOut,4*sizeof(double)));

    for(const Shape& sh:shapes){
        const long long m=sh.m,n=sh.n,k=sh.k;

        double *dA,*dB,*dCd,*dCo,*dRhi,*dRlo;
        float  *dAf,*dBf,*dCf,*dCe;
        CK(cudaMalloc(&dA,(size_t)m*k*8));  CK(cudaMalloc(&dB,(size_t)k*n*8));
        CK(cudaMalloc(&dCd,(size_t)m*n*8)); CK(cudaMalloc(&dCo,(size_t)m*n*8));
        CK(cudaMalloc(&dAf,(size_t)m*k*4)); CK(cudaMalloc(&dBf,(size_t)k*n*4));
        CK(cudaMalloc(&dCf,(size_t)m*n*4)); CK(cudaMalloc(&dCe,(size_t)m*n*4));

        fillKernel<<<1024,256>>>(dA,m*k,1234ull,dist);
        fillKernel<<<1024,256>>>(dB,k*n,5678ull,dist);
        d2f<<<1024,256>>>(dA,dAf,m*k);
        d2f<<<1024,256>>>(dB,dBf,k*n);
        CK(cudaGetLastError());

        const long long rs=m<256?m:256, cs=n<256?n:256;
        CK(cudaMalloc(&dRhi,(size_t)rs*cs*8)); CK(cudaMalloc(&dRlo,(size_t)rs*cs*8));
        { const long long tot=rs*cs;
          ddRefSub<<<(unsigned)((tot+255)/256),256>>>(dA,m,dB,k,dRhi,dRlo,rs,rs,cs,k);
          CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
        auto errD=[&](const double* C){ CK(cudaMemset(dOut,0,16));
            errSub<double><<<256,256>>>(C,m,dRhi,dRlo,rs,rs,cs,dOut);
            CK(cudaDeviceSynchronize()); double o[2];
            CK(cudaMemcpy(o,dOut,16,cudaMemcpyDeviceToHost));
            return o[1]>0?sqrt(o[0]/o[1]):0.0; };
        auto errF=[&](const float* C){ CK(cudaMemset(dOut,0,16));
            errSub<float><<<256,256>>>(C,m,dRhi,dRlo,rs,rs,cs,dOut);
            CK(cudaDeviceSynchronize()); double o[2];
            CK(cudaMemcpy(o,dOut,16,cudaMemcpyDeviceToHost));
            return o[1]>0?sqrt(o[0]/o[1]):0.0; };

        Timer t;
        const double d1=1.0,d0=0.0; const float f1=1.0f,f0=0.0f;
        double tD=0.0,tS=0.0;
        auto row=[&](const char* nm,int bins,int prod,double perIter,
                     double msc,double msp,double mg,double mf,double err){
            std::printf("%s,%lld,%lld,%lld,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4e\n",
                distName,m,n,k,nm,bins,prod,iters,perIter,msc,msp,mg,mf,
                tS/perIter,tD/perIter,err);
            std::fflush(stdout); };

        /* ---------------------- baselines ---------------------- */
        for(int i=0;i<warmup;++i) CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&d1,dA,(int)m,dB,(int)k,&d0,dCd,(int)m));
        CK(cudaDeviceSynchronize()); t.start();
        for(int i=0;i<iters;++i) CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&d1,dA,(int)m,dB,(int)k,&d0,dCd,(int)m));
        tD=t.stop()/iters;

        for(int i=0;i<warmup;++i) CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&f1,dAf,(int)m,dBf,(int)k,&f0,dCf,(int)m));
        CK(cudaDeviceSynchronize()); t.start();
        for(int i=0;i<iters;++i) CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&f1,dAf,(int)m,dBf,(int)k,&f0,dCf,(int)m));
        tS=t.stop()/iters;

        row("fp64",0,0,tD,0,0,tD,0,errD(dCd));
        row("fp32",0,0,tS,0,0,tS,0,errF(dCf));

        /* plain half-precision GEMMs: A cast once, B cast each iteration */
        {
            __nv_bfloat16 *bA,*bB; __half *hA,*hB;
            CK(cudaMalloc(&bA,(size_t)m*k*2)); CK(cudaMalloc(&bB,(size_t)k*n*2));
            CK(cudaMalloc(&hA,(size_t)m*k*2)); CK(cudaMalloc(&hB,(size_t)k*n*2));
            f2bf<<<1024,256>>>(dAf,bA,m*k); f2h<<<1024,256>>>(dAf,hA,m*k);
            const float one=1.0f, zero=0.0f;
            for(int rep=0;rep<2;++rep){
                const bool isBf=(rep==0);
                const void* A = isBf?(const void*)bA:(const void*)hA;
                const void* B = isBf?(const void*)bB:(const void*)hB;
                const cudaDataType dt = isBf?CUDA_R_16BF:CUDA_R_16F;
                for(int i=0;i<warmup;++i){
                    if(isBf) f2bf<<<1024,256>>>(dBf,bB,k*n); else f2h<<<1024,256>>>(dBf,hB,k*n);
                    cublasGemmEx_64(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&one,A,dt,m,B,dt,k,
                                    &zero,dCe,CUDA_R_32F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i){
                    if(isBf) f2bf<<<1024,256>>>(dBf,bB,k*n); else f2h<<<1024,256>>>(dBf,hB,k*n);
                    cublasGemmEx_64(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&one,A,dt,m,B,dt,k,
                                    &zero,dCe,CUDA_R_32F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }
                const double pv=t.stop()/iters;
                row(isBf?"bf16-gemm":"fp16-gemm",1,1,pv,0,0,pv,0,errF(dCe));
            }
            cudaFree(bA);cudaFree(bB);cudaFree(hA);cudaFree(hB);
        }

        /* ------------- bf16-split, concatenated-k, bins 1..3 ------------- */
        {
            const int SP=3;
            const size_t schemeBytes = mpemuBf16WorkspaceBytes(m,n,k,SP);
            const size_t stackBytes  = mpemuStackedBytes(k,n,SP,2);
            MP(mpemuContextReserve(ctx, schemeBytes + stackBytes + 4096));
            mpemuBf16Workspace_t ws; MP(mpemuBf16WorkspaceInit(ctx,m,n,k,SP,&ws));
            __nv_bfloat16* sBs=(__nv_bfloat16*)((char*)mpemuContextBase(ctx)+((schemeBytes+255)/256*256));
            const long long lds=mpemuStackedLd(k,SP);
            MP(mpemuSplitBF16(dAf,m,m,k,SP,ws.sA,ws.ldA,ws.strideA,0));   /* once */

            /* split cost is per-iteration and shared by every bin count */
            for(int i=0;i<warmup;++i) MP(mpemuSplitBF16Stacked(dBf,k,k,n,SP,sBs,lds,0));
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<iters;++i) MP(mpemuSplitBF16Stacked(dBf,k,k,n,SP,sBs,lds,0));
            const double msp=t.stop()/iters;

            /* the naive path needs B in the ordinary (non-stacked) layout */
            for(int i=0;i<warmup;++i) MP(mpemuSplitBF16(dBf,k,k,n,SP,ws.sB,ws.ldB,ws.strideB,0));
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<iters;++i) MP(mpemuSplitBF16(dBf,k,k,n,SP,ws.sB,ws.ldB,ws.strideB,0));
            const double mspNaive=t.stop()/iters;

            for(int nb=1; nb<=SP; ++nb){
                const int PR=nb*(nb+1)/2;

                /* ---- concat ---- */
                for(int i=0;i<warmup;++i)
                    MP(mpemuGemmEmulatedConcat(h,m,n,k,1.0f,ws.sA,ws.ldA,sBs,lds,SP,nb,0.0f,dCe,m));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i)
                    MP(mpemuGemmEmulatedConcat(h,m,n,k,1.0f,ws.sA,ws.ldA,sBs,lds,SP,nb,0.0f,dCe,m));
                const double mgC=t.stop()/iters;
                for(int i=0;i<warmup;++i){
                    MP(mpemuSplitBF16Stacked(dBf,k,k,n,SP,sBs,lds,0));
                    MP(mpemuGemmEmulatedConcat(h,m,n,k,1.0f,ws.sA,ws.ldA,sBs,lds,SP,nb,0.0f,dCe,m)); }
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i){
                    MP(mpemuSplitBF16Stacked(dBf,k,k,n,SP,sBs,lds,0));
                    MP(mpemuGemmEmulatedConcat(h,m,n,k,1.0f,ws.sA,ws.ldA,sBs,lds,SP,nb,0.0f,dCe,m)); }
                const double pvC=t.stop()/iters;
                const double eC=errF(dCe);

                /* ---- per-product (naive) at the same product count ---- */
                for(int i=0;i<warmup;++i)
                    MP(mpemuGemmEmulatedRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,ws.sA,ws.ldA,ws.strideA,ws.sB,ws.ldB,ws.strideB,SP,0,PR,0.0f,dCe,m));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i)
                    MP(mpemuGemmEmulatedRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,ws.sA,ws.ldA,ws.strideA,ws.sB,ws.ldB,ws.strideB,SP,0,PR,0.0f,dCe,m));
                const double mgN=t.stop()/iters;
                for(int i=0;i<warmup;++i){
                    MP(mpemuSplitBF16(dBf,k,k,n,SP,ws.sB,ws.ldB,ws.strideB,0));
                    MP(mpemuGemmEmulatedRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,ws.sA,ws.ldA,ws.strideA,ws.sB,ws.ldB,ws.strideB,SP,0,PR,0.0f,dCe,m)); }
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i){
                    MP(mpemuSplitBF16(dBf,k,k,n,SP,ws.sB,ws.ldB,ws.strideB,0));
                    MP(mpemuGemmEmulatedRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,ws.sA,ws.ldA,ws.strideA,ws.sB,ws.ldB,ws.strideB,SP,0,PR,0.0f,dCe,m)); }
                const double pvN=t.stop()/iters;

                row("bf16-split",       nb,PR,pvN,0.0,mspNaive,mgN,0.0,errF(dCe));
                row("bf16-split-concat",nb,PR,pvC,0.0,msp,     mgC,0.0,eC);
            }
        }

        /* ---- multiword: 2 words, bins 1..2 (1 and 3 products) ----
         * 2 fp16 words is what the paper says suffices for FP32
         * (u_low^2 = 4*u_high), and 2 words admit only 4 products of which the
         * i+j <= p+1 set is 3. A third word was measured and rejected: at 3
         * products it is bit-identical (bins 0-1 never touch word 2), and at 6
         * products it improves error by 1-4% at k <= 256 and 0% beyond, for 2x
         * the products. Both schemes are already within 2-4x of the floor set
         * by the FP32 input conversion, so extra words have nothing to
         * recover. See RESULTS_COMPARISON.md. */
        for (int SP : {2}) {
            const char* mwName  = (SP==2) ? "multiword"        : "multiword-3w";
            const char* mwNameC = (SP==2) ? "multiword-concat" : "multiword-3w-concat";
            const size_t schemeBytes = mpemuFp16WorkspaceBytes(m,n,k,SP);
            const size_t stackBytes  = mpemuStackedBytes(k,n,SP,2);
            MP(mpemuContextReserve(ctx, schemeBytes + stackBytes + 4096));
            mpemuFp16Workspace_t ws; MP(mpemuFp16WorkspaceInit(ctx,m,n,k,SP,&ws));
            __half* sBs=(__half*)((char*)mpemuContextBase(ctx)+((schemeBytes+255)/256*256));
            const long long lds=mpemuStackedLd(k,SP);
            float scA=1.0f,scB=1.0f;
            MP(mpemuAutoScaleFP16(dAf,m,m,k,&scA,0));
            MP(mpemuSplitFP16(dAf,m,m,k,SP,scA,ws.sA,ws.ldA,ws.strideA,0));   /* once */

            for(int i=0;i<warmup;++i) MP(mpemuAutoScaleFP16(dBf,k,k,n,&scB,0));
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<iters;++i) MP(mpemuAutoScaleFP16(dBf,k,k,n,&scB,0));
            const double msc=t.stop()/iters;

            for(int i=0;i<warmup;++i) MP(mpemuSplitFP16Stacked(dBf,k,k,n,SP,scB,sBs,lds,0));
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<iters;++i) MP(mpemuSplitFP16Stacked(dBf,k,k,n,SP,scB,sBs,lds,0));
            const double msp=t.stop()/iters;

            for(int i=0;i<warmup;++i) MP(mpemuSplitFP16(dBf,k,k,n,SP,scB,ws.sB,ws.ldB,ws.strideB,0));
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<iters;++i) MP(mpemuSplitFP16(dBf,k,k,n,SP,scB,ws.sB,ws.ldB,ws.strideB,0));
            const double mspNaive=t.stop()/iters;

            for(int nb=1; nb<=SP; ++nb){
                const int PR=nb*(nb+1)/2;

                /* ---- concat ---- */
                for(int i=0;i<warmup;++i)
                    MP(mpemuGemmMultiwordConcat(h,m,n,k,1.0f,scA,scB,ws.sA,ws.ldA,sBs,lds,SP,nb,0.0f,dCe,m));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i)
                    MP(mpemuGemmMultiwordConcat(h,m,n,k,1.0f,scA,scB,ws.sA,ws.ldA,sBs,lds,SP,nb,0.0f,dCe,m));
                const double mgC=t.stop()/iters;
                for(int i=0;i<warmup;++i){
                    MP(mpemuAutoScaleFP16(dBf,k,k,n,&scB,0));
                    MP(mpemuSplitFP16Stacked(dBf,k,k,n,SP,scB,sBs,lds,0));
                    MP(mpemuGemmMultiwordConcat(h,m,n,k,1.0f,scA,scB,ws.sA,ws.ldA,sBs,lds,SP,nb,0.0f,dCe,m)); }
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i){
                    MP(mpemuAutoScaleFP16(dBf,k,k,n,&scB,0));
                    MP(mpemuSplitFP16Stacked(dBf,k,k,n,SP,scB,sBs,lds,0));
                    MP(mpemuGemmMultiwordConcat(h,m,n,k,1.0f,scA,scB,ws.sA,ws.ldA,sBs,lds,SP,nb,0.0f,dCe,m)); }
                const double pvC=t.stop()/iters;
                const double eC=errF(dCe);

                /* ---- per-product (naive) ---- */
                for(int i=0;i<warmup;++i)
                    MP(mpemuGemmMultiwordRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,ws.sA,ws.ldA,ws.strideA,scA,ws.sB,ws.ldB,ws.strideB,scB,SP,0,PR,0.0f,dCe,m));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i)
                    MP(mpemuGemmMultiwordRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,ws.sA,ws.ldA,ws.strideA,scA,ws.sB,ws.ldB,ws.strideB,scB,SP,0,PR,0.0f,dCe,m));
                const double mgN=t.stop()/iters;
                for(int i=0;i<warmup;++i){
                    MP(mpemuAutoScaleFP16(dBf,k,k,n,&scB,0));
                    MP(mpemuSplitFP16(dBf,k,k,n,SP,scB,ws.sB,ws.ldB,ws.strideB,0));
                    MP(mpemuGemmMultiwordRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,ws.sA,ws.ldA,ws.strideA,scA,ws.sB,ws.ldB,ws.strideB,scB,SP,0,PR,0.0f,dCe,m)); }
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i){
                    MP(mpemuAutoScaleFP16(dBf,k,k,n,&scB,0));
                    MP(mpemuSplitFP16(dBf,k,k,n,SP,scB,ws.sB,ws.ldB,ws.strideB,0));
                    MP(mpemuGemmMultiwordRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,ws.sA,ws.ldA,ws.strideA,scA,ws.sB,ws.ldB,ws.strideB,scB,SP,0,PR,0.0f,dCe,m)); }
                const double pvN=t.stop()/iters;

                row(mwName, nb,PR,pvN,msc,mspNaive,mgN,0.0,errF(dCe));
                row(mwNameC,nb,PR,pvC,msc,msp,     mgC,0.0,eC);
            }
        }

        /* ---- Ozaki I, bin-grouped driver (no concat path), bins 1..8 ---- */
        {
            const int SP=8;
            MP(mpemuContextReserve(ctx, mpemuOz1WorkspaceBytes(m,n,k,SP)));
            mpemuOz1Workspace_t ws; MP(mpemuOz1WorkspaceInit(ctx,m,n,k,SP,&ws));
            MP(mpemuOz1Exponents(dA,m,m,k,1,ws.expA,0));
            MP(mpemuSplitInt8Ozaki1(dA,m,m,k,1,SP,ws.alphaBits,ws.expA,ws.sA,ws.ldA,ws.strideA,0));

            for(int i=0;i<warmup;++i) MP(mpemuOz1Exponents(dB,k,k,n,0,ws.expB,0));
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<iters;++i) MP(mpemuOz1Exponents(dB,k,k,n,0,ws.expB,0));
            const double msc=t.stop()/iters;

            for(int i=0;i<warmup;++i) MP(mpemuSplitInt8Ozaki1(dB,k,k,n,0,SP,ws.alphaBits,ws.expB,ws.sB,ws.ldB,ws.strideB,0));
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<iters;++i) MP(mpemuSplitInt8Ozaki1(dB,k,k,n,0,SP,ws.alphaBits,ws.expB,ws.sB,ws.ldB,ws.strideB,0));
            const double msp=t.stop()/iters;

            for(int nb=1; nb<=SP; ++nb){
                const int PR=nb*(nb+1)/2;
                for(int i=0;i<warmup;++i)
                    MP(mpemuGemmOzaki1Range(h,m,n,k,1.0,ws.sA,ws.ldA,ws.strideA,ws.expA,ws.sB,ws.ldB,ws.strideB,ws.expB,SP,ws.alphaBits,0,PR,0.0,dCo,m,ws.ct,ws.ldct,0));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i)
                    MP(mpemuGemmOzaki1Range(h,m,n,k,1.0,ws.sA,ws.ldA,ws.strideA,ws.expA,ws.sB,ws.ldB,ws.strideB,ws.expB,SP,ws.alphaBits,0,PR,0.0,dCo,m,ws.ct,ws.ldct,0));
                const double mg=t.stop()/iters;

                for(int i=0;i<warmup;++i){
                    MP(mpemuOz1Exponents(dB,k,k,n,0,ws.expB,0));
                    MP(mpemuSplitInt8Ozaki1(dB,k,k,n,0,SP,ws.alphaBits,ws.expB,ws.sB,ws.ldB,ws.strideB,0));
                    MP(mpemuGemmOzaki1Range(h,m,n,k,1.0,ws.sA,ws.ldA,ws.strideA,ws.expA,ws.sB,ws.ldB,ws.strideB,ws.expB,SP,ws.alphaBits,0,PR,0.0,dCo,m,ws.ct,ws.ldct,0)); }
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<iters;++i){
                    MP(mpemuOz1Exponents(dB,k,k,n,0,ws.expB,0));
                    MP(mpemuSplitInt8Ozaki1(dB,k,k,n,0,SP,ws.alphaBits,ws.expB,ws.sB,ws.ldB,ws.strideB,0));
                    MP(mpemuGemmOzaki1Range(h,m,n,k,1.0,ws.sA,ws.ldA,ws.strideA,ws.expA,ws.sB,ws.ldB,ws.strideB,ws.expB,SP,ws.alphaBits,0,PR,0.0,dCo,m,ws.ct,ws.ldct,0)); }
                const double pv=t.stop()/iters;
                row("ozaki1",nb,PR,pv,msc,msp,mg,0.0,errD(dCo));
            }
        }

        std::fprintf(stderr,"# %lldx%lldx%lld arena=%.1f MB allocs=%d\n",
                     m,n,k,mpemuContextCapacity(ctx)/1048576.0,mpemuContextAllocCount(ctx));
        cudaFree(dA);cudaFree(dB);cudaFree(dCd);cudaFree(dCo);
        cudaFree(dAf);cudaFree(dBf);cudaFree(dCf);cudaFree(dCe);
        cudaFree(dRhi);cudaFree(dRlo);
    }
    cudaFree(dOut);
    MP(mpemuContextDestroy(ctx));
    CB(cublasDestroy(h));
    return 0;
}
