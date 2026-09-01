/* All three emulation schemes on one footing, against cublasDgemm and
 * cublasSgemm, over two input distributions and a fixed set of shapes.
 *
 * Each scheme is split ONCE, as many times as it needs to reach full accuracy
 * for its target format, and the x axis is then the NUMBER OF PRODUCTS
 * accumulated into the output:
 *
 *   bf16   (Henry)   3 splits  ->  FP32   (3 x ~9 bits >= 24);  1..9 products
 *   fp16mw (Fasi)    2 splits  ->  FP32   (2 x ~12 bits = 24);  1..4 products
 *   ozaki1 (Ootomo)  8 slices  ->  FP64   (8 x 7 bits >= 53);   1..36 products
 *
 * Products are added one at a time in ascending-bin order, exactly as a
 * caller would refine a result, and the error is measured after every single
 * addition. Timing is likewise per-step, so the reported cost at p products
 * is the cumulative cost of having refined that far -- and is split into the
 * one-off costs (scaling, splitting) and the per-product costs (GEMM, and for
 * Ozaki I the INT32->FP64 fold), so the fixed overhead is visible separately.
 *
 * bf16 and multiword fp16 consume FP32, so the FP32 line is their error
 * FLOOR, not merely a reference. Ozaki I consumes FP64 directly.
 *
 * bf16 and multiword fp16 consume FP32, so their error floor is the FP32
 * rounding of the input; Ozaki I consumes the FP64 input directly. All errors
 * are measured against a double-double reference built from the FP64 data.
 *
 * The reference is computed on a leading sub-block of C (at most 256x256) but
 * with the FULL inner dimension, so every sampled element is a complete dot
 * product. That keeps a ~106-bit reference affordable even at 16384^3.
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

/* ---------------------------------------------------------------- data --- */

__device__ __forceinline__ unsigned long long sm64(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x>>27)) * 0x94D049BB133111EBull;
    return x ^ (x>>31);
}
__device__ __forceinline__ double u01(unsigned long long h)
{
    return (double)(h>>11) * (1.0/9007199254740992.0);
}

/* dist 0: uniform on [-1,1].  dist 1: standard normal (Box-Muller). */
__global__ void fillKernel(double* a, long long n, unsigned long long seed, int dist)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st){
        const unsigned long long h1=sm64(seed ^ (unsigned long long)i);
        if(dist==0){
            a[i] = 2.0*u01(h1) - 1.0;
        } else {
            const unsigned long long h2=sm64(h1 ^ 0x9E3779B97F4A7C15ull);
            double v1=u01(h1); if(v1<1e-300) v1=1e-300;
            const double v2=u01(h2);
            a[i] = sqrt(-2.0*log(v1)) * cos(6.283185307179586*v2);
        }
    }
}
__global__ void d2f(const double* __restrict__ s, float* __restrict__ f, long long n)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st) f[i]=(float)s[i];
}

/* ------------------------------------------------- double-double reference */

__device__ __forceinline__ void ddAcc(double& sh,double& sl,double p,double e)
{
    double t=sh+p, bv=t-sh;
    sl += ((sh-(t-bv))+(p-bv)) + e;
    double s2=t+sl; sl=sl-(s2-t); sh=s2;
}
/* Leading rs x cs block of A*B, full inner dimension, ~106 bits. */
__global__ void ddRefSub(const double* __restrict__ A, long long lda,
                         const double* __restrict__ B, long long ldb,
                         double* __restrict__ Rhi, double* __restrict__ Rlo,
                         long long ldr, long long rs, long long cs, long long k)
{
    const long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= rs*cs) return;
    const long long r=idx%rs, c=idx/rs;
    double sh=0.0, sl=0.0;
    for(long long l=0;l<k;++l){
        const double a=A[l*lda+r], b=B[c*ldb+l];
        const double p=a*b, e=fma(a,b,-p);      /* exact product */
        ddAcc(sh,sl,p,e);
    }
    Rhi[c*ldr+r]=sh; Rlo[c*ldr+r]=sl;
}

template<typename T>
__global__ void errSub(const T* __restrict__ C, long long ldc,
                       const double* __restrict__ Rhi, const double* __restrict__ Rlo,
                       long long ldr, long long rs, long long cs, double* out)
{
    __shared__ double sd[256], sr[256];
    double ad=0, ar=0;
    const long long total=rs*cs;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long st=(long long)gridDim.x*blockDim.x;
    for(; i<total; i+=st){
        const long long r=i%rs, c=i/rs;
        const double hi=Rhi[c*ldr+r], lo=Rlo[c*ldr+r];
        const double d=(((double)C[c*ldc+r]-hi)-lo), ref=hi+lo;
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

/* one scheme's fixed configuration */
struct Scheme {
    const char* name;
    int splits;        /* fixed: enough for full accuracy in the target format */
    int products;      /* x-axis extent */
};

int main(int argc,char** argv)
{
    int dist=0, warmup=2, repsOverride=0;
    long long maxDim=0;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto val=[&]{ return (i+1<argc)?argv[++i]:nullptr; };
        if(a=="--dist"){ std::string d=val(); dist=(d=="gaussian")?1:0; }
        else if(a=="--reps") repsOverride=atoi(val());
        else if(a=="--max-dim") maxDim=atoll(val());
        else if(a=="--help"){ std::printf("usage: %s [--dist uniform|gaussian] [--reps N] [--max-dim N]\n",argv[0]); return 0; }
    }
    const char* distName = dist? "gaussian" : "uniform";

    std::vector<Shape> shapes={
        {256,256,256},{1024,1024,1024},{2048,2048,2048},
        {16384,16384,16384},{16384,256,1000},{16384,64,100}};
    if(maxDim>0){ std::vector<Shape> keep;
        for(auto&s:shapes) if(s.m<=maxDim&&s.n<=maxDim&&s.k<=maxDim) keep.push_back(s);
        shapes.swap(keep); }

    const Scheme BF16 {"bf16",   3, 9};
    const Scheme FP16 {"fp16mw", 2, 4};
    const Scheme OZ1  {"ozaki1", 8, 36};

    cudaDeviceProp prop; int dev=0;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop,dev));
    std::fprintf(stderr,"# device %s  dist=%s\n",prop.name,distName);

    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));

    std::printf("dist,m,n,k,scheme,splits,products,ms_scale,ms_split,"
                "ms_gemm_cum,ms_fold_cum,ms_total,rel_err\n");

    double* dOut; CK(cudaMalloc(&dOut,4*sizeof(double)));

    for(const Shape& sh:shapes){
        const long long m=sh.m,n=sh.n,k=sh.k;
        const double flops=2.0*(double)m*(double)n*(double)k;
        const int reps = repsOverride>0 ? repsOverride
                                        : (int)fmin(10.0, fmax(2.0, 2e12/flops));

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

        const long long rs = m<256?m:256, cs = n<256?n:256;
        CK(cudaMalloc(&dRhi,(size_t)rs*cs*8)); CK(cudaMalloc(&dRlo,(size_t)rs*cs*8));
        { const long long tot=rs*cs;
          ddRefSub<<<(unsigned)((tot+255)/256),256>>>(dA,m,dB,k,dRhi,dRlo,rs,rs,cs,k);
          CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }

        auto errD=[&](const double* C)->double{
            CK(cudaMemset(dOut,0,2*sizeof(double)));
            errSub<double><<<256,256>>>(C,m,dRhi,dRlo,rs,rs,cs,dOut);
            CK(cudaDeviceSynchronize()); double o[2];
            CK(cudaMemcpy(o,dOut,2*sizeof(double),cudaMemcpyDeviceToHost));
            return o[1]>0?sqrt(o[0]/o[1]):0.0; };
        auto errF=[&](const float* C)->double{
            CK(cudaMemset(dOut,0,2*sizeof(double)));
            errSub<float><<<256,256>>>(C,m,dRhi,dRlo,rs,rs,cs,dOut);
            CK(cudaDeviceSynchronize()); double o[2];
            CK(cudaMemcpy(o,dOut,2*sizeof(double),cudaMemcpyDeviceToHost));
            return o[1]>0?sqrt(o[0]/o[1]):0.0; };

        Timer t;
        const double d1=1.0,d0=0.0; const float f1=1.0f,f0=0.0f;
        auto row=[&](const char* sc,int sp,int pr,double msc,double msp,
                     double mg,double mf,double err){
            std::printf("%s,%lld,%lld,%lld,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.4e\n",
                        distName,m,n,k,sc,sp,pr,msc,msp,mg,mf,msc+msp+mg+mf,err);
            std::fflush(stdout); };

        /* ------------------------- baselines ------------------------- */
        for(int i=0;i<warmup;++i) CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&d1,dA,(int)m,dB,(int)k,&d0,dCd,(int)m));
        CK(cudaDeviceSynchronize()); t.start();
        for(int i=0;i<reps;++i) CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&d1,dA,(int)m,dB,(int)k,&d0,dCd,(int)m));
        const double dms=t.stop()/reps;
        row("fp64",0,0,0.0,0.0,dms,0.0,errD(dCd));

        for(int i=0;i<warmup;++i) CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&f1,dAf,(int)m,dBf,(int)k,&f0,dCf,(int)m));
        CK(cudaDeviceSynchronize()); t.start();
        for(int i=0;i<reps;++i) CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&f1,dAf,(int)m,dBf,(int)k,&f0,dCf,(int)m));
        const double sms=t.stop()/reps;
        row("fp32",0,0,0.0,0.0,sms,0.0,errF(dCf));

        mpemuTerm_t terms[64];

        /* ------------------------- bf16 ------------------------- */
        {
            const Scheme& S=BF16;
            const long long lda=mpemuSplitLd(m), sA=lda*k;
            const long long ldb=mpemuSplitLd(k), sB=ldb*n;
            __nv_bfloat16 *dSA,*dSB;
            CK(cudaMalloc(&dSA,(size_t)sA*S.splits*2));
            CK(cudaMalloc(&dSB,(size_t)sB*S.splits*2));

            for(int i=0;i<warmup;++i){
                MP(mpemuSplitBF16(dAf,m,m,k,S.splits,dSA,lda,sA,0));
                MP(mpemuSplitBF16(dBf,k,k,n,S.splits,dSB,ldb,sB,0)); }
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<reps;++i){
                MP(mpemuSplitBF16(dAf,m,m,k,S.splits,dSA,lda,sA,0));
                MP(mpemuSplitBF16(dBf,k,k,n,S.splits,dSB,ldb,sB,0)); }
            const double msplit=t.stop()/reps;

            /* per-product timing */
            std::vector<double> step(S.products,0.0);
            for(int p=0;p<S.products;++p){
                for(int i=0;i<warmup;++i)
                    MP(mpemuGemmEmulatedRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,dSA,lda,sA,dSB,ldb,sB,S.splits,p,p+1,1.0f,dCe,m));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<reps;++i)
                    MP(mpemuGemmEmulatedRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,dSA,lda,sA,dSB,ldb,sB,S.splits,p,p+1,1.0f,dCe,m));
                step[p]=t.stop()/reps;
            }
            /* accuracy: refine one product at a time */
            double cum=0.0;
            for(int p=0;p<S.products;++p){
                MP(mpemuGemmEmulatedRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,dSA,lda,sA,dSB,ldb,sB,
                                          S.splits,p,p+1,(p==0)?0.0f:1.0f,dCe,m));
                CK(cudaDeviceSynchronize());
                cum+=step[p];
                row(S.name,S.splits,p+1,0.0,msplit,cum,0.0,errF(dCe));
            }
            cudaFree(dSA); cudaFree(dSB);
        }

        /* ------------------------- multiword fp16 ------------------------- */
        {
            const Scheme& S=FP16;
            const long long lda=mpemuSplitLd(m), sA=lda*k;
            const long long ldb=mpemuSplitLd(k), sB=ldb*n;
            __half *dSA,*dSB;
            CK(cudaMalloc(&dSA,(size_t)sA*S.splits*2));
            CK(cudaMalloc(&dSB,(size_t)sB*S.splits*2));

            float scA=1.0f,scB=1.0f;
            for(int i=0;i<warmup;++i){
                MP(mpemuAutoScaleFP16(dAf,m,m,k,&scA,0));
                MP(mpemuAutoScaleFP16(dBf,k,k,n,&scB,0)); }
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<reps;++i){
                MP(mpemuAutoScaleFP16(dAf,m,m,k,&scA,0));
                MP(mpemuAutoScaleFP16(dBf,k,k,n,&scB,0)); }
            const double mscale=t.stop()/reps;

            for(int i=0;i<warmup;++i){
                MP(mpemuSplitFP16(dAf,m,m,k,S.splits,scA,dSA,lda,sA,0));
                MP(mpemuSplitFP16(dBf,k,k,n,S.splits,scB,dSB,ldb,sB,0)); }
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<reps;++i){
                MP(mpemuSplitFP16(dAf,m,m,k,S.splits,scA,dSA,lda,sA,0));
                MP(mpemuSplitFP16(dBf,k,k,n,S.splits,scB,dSB,ldb,sB,0)); }
            const double msplit=t.stop()/reps;

            std::vector<double> step(S.products,0.0);
            for(int p=0;p<S.products;++p){
                for(int i=0;i<warmup;++i)
                    MP(mpemuGemmMultiwordRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,dSA,lda,sA,scA,dSB,ldb,sB,scB,S.splits,p,p+1,1.0f,dCe,m));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<reps;++i)
                    MP(mpemuGemmMultiwordRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,dSA,lda,sA,scA,dSB,ldb,sB,scB,S.splits,p,p+1,1.0f,dCe,m));
                step[p]=t.stop()/reps;
            }
            double cum=0.0;
            for(int p=0;p<S.products;++p){
                MP(mpemuGemmMultiwordRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,dSA,lda,sA,scA,dSB,ldb,sB,scB,
                                           S.splits,p,p+1,(p==0)?0.0f:1.0f,dCe,m));
                CK(cudaDeviceSynchronize());
                cum+=step[p];
                row(S.name,S.splits,p+1,mscale,msplit,cum,0.0,errF(dCe));
            }
            cudaFree(dSA); cudaFree(dSB);
        }

        /* ------------------------- Ozaki I on INT8 ------------------------- */
        {
            const Scheme& S=OZ1;
            const int alpha=mpemuOz1BitsPerSlice(k);
            const long long lda=mpemuOz1SplitLd(k), sA=lda*m;
            const long long ldb=mpemuOz1SplitLd(k), sB=ldb*n;
            signed char *dSA,*dSB; int *dEA,*dEB,*dCt;
            CK(cudaMalloc(&dSA,(size_t)sA*S.splits)); CK(cudaMalloc(&dSB,(size_t)sB*S.splits));
            CK(cudaMalloc(&dEA,m*4)); CK(cudaMalloc(&dEB,n*4));
            CK(cudaMalloc(&dCt,(size_t)m*n*4));

            for(int i=0;i<warmup;++i){
                MP(mpemuOz1Exponents(dA,m,m,k,1,dEA,0));
                MP(mpemuOz1Exponents(dB,k,k,n,0,dEB,0)); }
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<reps;++i){
                MP(mpemuOz1Exponents(dA,m,m,k,1,dEA,0));
                MP(mpemuOz1Exponents(dB,k,k,n,0,dEB,0)); }
            const double mscale=t.stop()/reps;

            for(int i=0;i<warmup;++i){
                MP(mpemuSplitInt8Ozaki1(dA,m,m,k,1,S.splits,alpha,dEA,dSA,lda,sA,0));
                MP(mpemuSplitInt8Ozaki1(dB,k,k,n,0,S.splits,alpha,dEB,dSB,ldb,sB,0)); }
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<reps;++i){
                MP(mpemuSplitInt8Ozaki1(dA,m,m,k,1,S.splits,alpha,dEA,dSA,lda,sA,0));
                MP(mpemuSplitInt8Ozaki1(dB,k,k,n,0,S.splits,alpha,dEB,dSB,ldb,sB,0)); }
            const double msplit=t.stop()/reps;

            const int nt=mpemuTermSchedule(S.splits,S.products,terms);
            if(nt<S.products){ std::fprintf(stderr,"schedule short: %d\n",nt); return 1; }

            /* per-product timing, GEMM and fold measured apart */
            std::vector<double> sg(S.products,0.0), sf(S.products,0.0);
            for(int p=0;p<S.products;++p){
                const int bin=terms[p].i+terms[p].j;
                for(int i=0;i<warmup;++i)
                    MP(mpemuGemmInt8(h,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,1,
                        mpemuOz1PlaneConst(dSA,sA,terms[p].i),lda,
                        mpemuOz1PlaneConst(dSB,sB,terms[p].j),ldb,0,dCt,m));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<reps;++i)
                    MP(mpemuGemmInt8(h,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,1,
                        mpemuOz1PlaneConst(dSA,sA,terms[p].i),lda,
                        mpemuOz1PlaneConst(dSB,sB,terms[p].j),ldb,0,dCt,m));
                sg[p]=t.stop()/reps;

                for(int i=0;i<warmup;++i)
                    MP(mpemuOz1Accumulate(dCt,m,m,n,bin,alpha,dEA,dEB,1.0,1.0,dCo,m,0));
                CK(cudaDeviceSynchronize()); t.start();
                for(int i=0;i<reps;++i)
                    MP(mpemuOz1Accumulate(dCt,m,m,n,bin,alpha,dEA,dEB,1.0,1.0,dCo,m,0));
                sf[p]=t.stop()/reps;
            }
            /* accuracy: refine one product at a time */
            double cg=0.0, cf=0.0;
            for(int p=0;p<S.products;++p){
                const int bin=terms[p].i+terms[p].j;
                MP(mpemuGemmInt8(h,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,1,
                    mpemuOz1PlaneConst(dSA,sA,terms[p].i),lda,
                    mpemuOz1PlaneConst(dSB,sB,terms[p].j),ldb,0,dCt,m));
                MP(mpemuOz1Accumulate(dCt,m,m,n,bin,alpha,dEA,dEB,1.0,(p==0)?0.0:1.0,dCo,m,0));
                CK(cudaDeviceSynchronize());
                cg+=sg[p]; cf+=sf[p];
                row(S.name,S.splits,p+1,mscale,msplit,cg,cf,errD(dCo));
            }
            cudaFree(dSA);cudaFree(dSB);cudaFree(dEA);cudaFree(dEB);cudaFree(dCt);
        }

        std::fprintf(stderr,"# %lldx%lldx%lld reps=%d ref=%lldx%lld\n",m,n,k,reps,rs,cs);
        cudaFree(dA);cudaFree(dB);cudaFree(dCd);cudaFree(dCo);
        cudaFree(dAf);cudaFree(dBf);cudaFree(dCf);cudaFree(dCe);
        cudaFree(dRhi);cudaFree(dRlo);
    }
    cudaFree(dOut); CB(cublasDestroy(h));
    return 0;
}
