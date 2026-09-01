/* Ozaki scheme I on INT8 tensor cores vs cublasDgemm and cublasSgemm.
 *
 * Forward error for every method -- dgemm included -- is measured against a
 * double-double reference (exact fma-based products, ~106-bit accumulation),
 * since dgemm is itself the thing under test here.
 *
 * cublasSgemm is timed and scored on FP32 copies of the same FP64 inputs, so
 * its error legitimately includes the input rounding: that is what you would
 * actually pay by "just using sgemm".
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

#define CK(x) do { cudaError_t ck_=(x); if(ck_!=cudaSuccess){                      \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(ck_)); \
    std::exit(1);} } while(0)
#define CB(x) do { cublasStatus_t cb_=(x); if(cb_!=CUBLAS_STATUS_SUCCESS){         \
    std::fprintf(stderr,"%s:%d cublas %d\n",__FILE__,__LINE__,(int)cb_);         \
    std::exit(1);} } while(0)
#define MP(x) do { mpemuStatus_t mp_=(x); if(mp_!=MPEMU_STATUS_SUCCESS){           \
    std::fprintf(stderr,"%s:%d mpemu %s\n",__FILE__,__LINE__,                  \
    mpemuStatusString(mp_)); std::exit(1);} } while(0)

__device__ __forceinline__ unsigned long long sm64(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x>>30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x>>27)) * 0x94D049BB133111EBull;
    return x ^ (x>>31);
}
__global__ void fill(double* a, long long n, unsigned long long seed, double phi)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st){
        unsigned long long h=sm64(seed^(unsigned long long)i);
        double u=(double)(h>>11)*(1.0/9007199254740992.0);
        double v=u-0.5;
        if(phi>0.0){
            unsigned long long g=sm64(0xABCDEFull^seed^(unsigned long long)(i*2654435761ull));
            double w=(double)(g>>11)*(1.0/9007199254740992.0);
            v*=exp(phi*(2.0*w-1.0)*3.0);      /* (rand-0.5)*exp(phi*randn)-like */
        }
        a[i]=v;
    }
}
__global__ void d2f(const double* __restrict__ s, float* __restrict__ f, long long n)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st) f[i]=(float)s[i];
}

#define TS 16
__device__ __forceinline__ void ddAcc(double& sh,double& sl,double p,double e)
{
    double t=sh+p,bv=t-sh;
    sl += ((sh-(t-bv))+(p-bv))+e;
    double s2=t+sl; sl=sl-(s2-t); sh=s2;
}
__global__ void ddRef(const double* __restrict__ A,long long lda,
                      const double* __restrict__ B,long long ldb,
                      double* __restrict__ Chi,double* __restrict__ Clo,
                      long long ldc,long long m,long long n,long long k)
{
    __shared__ double As[TS][TS+1],Bs[TS][TS+1];
    int tx=threadIdx.x,ty=threadIdx.y;
    long long row=(long long)blockIdx.x*TS+tx,col=(long long)blockIdx.y*TS+ty;
    double sh=0.0,sl=0.0;
    for(long long t=0;t<k;t+=TS){
        As[ty][tx]=(row<m&&t+ty<k)?A[(t+ty)*lda+row]:0.0;
        Bs[tx][ty]=(col<n&&t+tx<k)?B[col*ldb+(t+tx)]:0.0;
        __syncthreads();
#pragma unroll
        for(int i2=0;i2<TS;++i2){
            double a=As[i2][tx],b=Bs[i2][ty];
            double p=a*b,e=fma(a,b,-p);
            ddAcc(sh,sl,p,e);
        }
        __syncthreads();
    }
    if(row<m&&col<n){ Chi[col*ldc+row]=sh; Clo[col*ldc+row]=sl; }
}
template<typename T>
__global__ void errK(const T* __restrict__ C,long long ldc,
                     const double* __restrict__ Chi,const double* __restrict__ Clo,
                     long long ldr,long long rows,long long cols,double* out)
{
    __shared__ double sd[256],sr[256];
    double ad=0,ar=0;
    long long total=rows*cols;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<total; i+=st){
        long long r=i%rows,c=i/rows;
        double hi=Chi[c*ldr+r],lo=Clo[c*ldr+r];
        double d=(((double)C[c*ldc+r]-hi)-lo),ref=hi+lo;
        ad+=d*d; ar+=ref*ref;
    }
    int t=threadIdx.x; sd[t]=ad; sr[t]=ar; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s){sd[t]+=sd[t+s];sr[t]+=sr[t+s];} __syncthreads(); }
    if(t==0){ atomicAdd(&out[0],sd[0]); atomicAdd(&out[1],sr[0]); }
}

struct Timer{
    cudaEvent_t a,b;
    Timer(){cudaEventCreate(&a);cudaEventCreate(&b);} 
    ~Timer(){cudaEventDestroy(a);cudaEventDestroy(b);} 
    void start(){CK(cudaEventRecord(a));}
    float stop(){CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
                 float ms;CK(cudaEventElapsedTime(&ms,a,b));return ms;}
};
struct Shape{ long long m,n,k; const char* tag; };

int main(int argc,char** argv)
{
    int reps=5,warmup=2;
    long long maxDim=0;
    double refFlopCap=7.0e10;      /* dd reference is ~20x dgemm */
    double phi=0.0;
    bool squareOnly=false,noCheck=false;

    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto val=[&]{ return (i+1<argc)?argv[++i]:nullptr; };
        if(a=="--reps")reps=atoi(val());
        else if(a=="--max-dim")maxDim=atoll(val());
        else if(a=="--ref-flops")refFlopCap=atof(val());
        else if(a=="--phi")phi=atof(val());
        else if(a=="--square")squareOnly=true;
        else if(a=="--no-check")noCheck=true;
        else if(a=="--help"){ std::printf("usage: %s [--reps N] [--max-dim N] [--phi F] [--square] [--no-check]\n",argv[0]); return 0; }
    }

    std::vector<Shape> shapes={
        {1024,1024,1024,"square"},{2048,2048,2048,"square"},
        {4096,4096,4096,"square"},{8192,8192,8192,"square"},
        {16384,16384,16384,"square"}};
    if(!squareOnly){
        shapes.push_back({16384,16384,2048,"k-thin"});
        shapes.push_back({4096,4096,16384,"k-fat"});
    }
    if(maxDim>0){ std::vector<Shape> keep;
        for(auto&s:shapes) if(s.m<=maxDim&&s.n<=maxDim&&s.k<=maxDim) keep.push_back(s);
        shapes.swap(keep); }

    cudaDeviceProp prop; int dev=0;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop,dev));
    std::fprintf(stderr,"# device %s  reps=%d phi=%.2f\n",prop.name,reps,phi);

    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));

    std::printf("shape,m,n,k,method,slices,bits,gemms,ms_gemm,ms_total,tflops_gemm,"
                "vs_dgemm,vs_sgemm,rel_frob\n");

    const int SMAX=8;
    double* dOut; CK(cudaMalloc(&dOut,4*8));

    for(const Shape& sh:shapes){
        const long long m=sh.m,n=sh.n,k=sh.k;
        const double flops=2.0*(double)m*(double)n*(double)k;
        const int alpha=mpemuOz1BitsPerSlice(k);
        const long long ldsa=mpemuOz1SplitLd(k), strA=ldsa*m;   /* A^T planes: k x m */
        const long long ldsb=mpemuOz1SplitLd(k), strB=ldsb*n;   /* B planes:   k x n */

        double *dA,*dB,*dCd,*dCo,*dRhi=nullptr,*dRlo=nullptr;
        float *dAf,*dBf,*dCf;
        signed char *dSA,*dSB;
        int *dEA,*dEB,*dCt;
        CK(cudaMalloc(&dA,(size_t)m*k*8)); CK(cudaMalloc(&dB,(size_t)k*n*8));
        CK(cudaMalloc(&dCd,(size_t)m*n*8)); CK(cudaMalloc(&dCo,(size_t)m*n*8));
        CK(cudaMalloc(&dAf,(size_t)m*k*4)); CK(cudaMalloc(&dBf,(size_t)k*n*4));
        CK(cudaMalloc(&dCf,(size_t)m*n*4));
        CK(cudaMalloc(&dSA,(size_t)strA*SMAX)); CK(cudaMalloc(&dSB,(size_t)strB*SMAX));
        CK(cudaMalloc(&dEA,m*4)); CK(cudaMalloc(&dEB,n*4));
        CK(cudaMalloc(&dCt,(size_t)m*n*4));

        fill<<<1024,256>>>(dA,m*k,11ull,phi);
        fill<<<1024,256>>>(dB,k*n,22ull,phi);
        d2f<<<1024,256>>>(dA,dAf,m*k);
        d2f<<<1024,256>>>(dB,dBf,k*n);
        CK(cudaGetLastError());

        const bool doCheck = !noCheck && flops<=refFlopCap;
        if(doCheck){
            CK(cudaMalloc(&dRhi,(size_t)m*n*8)); CK(cudaMalloc(&dRlo,(size_t)m*n*8));
            dim3 blk(TS,TS),grd((unsigned)((m+TS-1)/TS),(unsigned)((n+TS-1)/TS));
            ddRef<<<grd,blk>>>(dA,m,dB,k,dRhi,dRlo,m,m,n,k);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        }
        char eb[40];
        auto fmt=[&](double e){ if(doCheck) std::snprintf(eb,sizeof eb,"%.3e",e);
                                else std::snprintf(eb,sizeof eb,""); return eb; };
        auto errD=[&](const double* C)->double{
            if(!doCheck) return 0.0;
            CK(cudaMemset(dOut,0,2*8)); errK<double><<<1024,256>>>(C,m,dRhi,dRlo,m,m,n,dOut);
            CK(cudaDeviceSynchronize()); double o[2];
            CK(cudaMemcpy(o,dOut,2*8,cudaMemcpyDeviceToHost));
            return o[1]>0?sqrt(o[0]/o[1]):0.0; };
        auto errF=[&](const float* C)->double{
            if(!doCheck) return 0.0;
            CK(cudaMemset(dOut,0,2*8)); errK<float><<<1024,256>>>(C,m,dRhi,dRlo,m,m,n,dOut);
            CK(cudaDeviceSynchronize()); double o[2];
            CK(cudaMemcpy(o,dOut,2*8,cudaMemcpyDeviceToHost));
            return o[1]>0?sqrt(o[0]/o[1]):0.0; };

        Timer t;
        const double d1=1.0,d0=0.0; const float f1=1.0f,f0=0.0f;

        for(int i=0;i<warmup;++i) CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&d1,dA,(int)m,dB,(int)k,&d0,dCd,(int)m));
        CK(cudaDeviceSynchronize()); t.start();
        for(int i=0;i<reps;++i) CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&d1,dA,(int)m,dB,(int)k,&d0,dCd,(int)m));
        const double dms=t.stop()/reps;

        for(int i=0;i<warmup;++i) CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&f1,dAf,(int)m,dBf,(int)k,&f0,dCf,(int)m));
        CK(cudaDeviceSynchronize()); t.start();
        for(int i=0;i<reps;++i) CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)m,(int)n,(int)k,&f1,dAf,(int)m,dBf,(int)k,&f0,dCf,(int)m));
        const double sms=t.stop()/reps;

        std::printf("%s,%lld,%lld,%lld,cublasDgemm,,,,%.4f,%.4f,%.2f,1.000,%.3f,%s\n",
            sh.tag,m,n,k,dms,dms,flops/(dms*1e9),sms/dms,fmt(errD(dCd)));
        std::printf("%s,%lld,%lld,%lld,cublasSgemm,,,,%.4f,%.4f,%.2f,%.3f,1.000,%s\n",
            sh.tag,m,n,k,sms,sms,flops/(sms*1e9),dms/sms,fmt(errF(dCf)));
        std::fflush(stdout);

        MP(mpemuOz1Exponents(dA,m,m,k,1,dEA,0));
        MP(mpemuOz1Exponents(dB,k,k,n,0,dEB,0));

        for(int s=2;s<=SMAX;++s){
            /* split cost, timed separately */
            for(int i=0;i<warmup;++i){
                MP(mpemuSplitInt8Ozaki1(dA,m,m,k,1,s,alpha,dEA,dSA,ldsa,strA,0));
                MP(mpemuSplitInt8Ozaki1(dB,k,k,n,0,s,alpha,dEB,dSB,ldsb,strB,0)); }
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<reps;++i){
                MP(mpemuSplitInt8Ozaki1(dA,m,m,k,1,s,alpha,dEA,dSA,ldsa,strA,0));
                MP(mpemuSplitInt8Ozaki1(dB,k,k,n,0,s,alpha,dEB,dSB,ldsb,strB,0)); }
            const double spl=t.stop()/reps;

            for(int i=0;i<warmup;++i)
                MP(mpemuGemmOzaki1(h,m,n,k,1.0,dSA,ldsa,strA,dEA,dSB,ldsb,strB,dEB,s,alpha,0.0,dCo,m,dCt,m,0));
            CK(cudaDeviceSynchronize()); t.start();
            for(int i=0;i<reps;++i)
                MP(mpemuGemmOzaki1(h,m,n,k,1.0,dSA,ldsa,strA,dEA,dSB,ldsb,strB,dEB,s,alpha,0.0,dCo,m,dCt,m,0));
            const double gms=t.stop()/reps;

            std::printf("%s,%lld,%lld,%lld,ozaki1,%d,%d,%d,%.4f,%.4f,%.2f,%.3f,%.3f,%s\n",
                sh.tag,m,n,k,s,s*alpha,mpemuOz1Macs(s),gms,gms+spl,
                flops/(gms*1e9),dms/gms,sms/gms,fmt(errD(dCo)));
            std::fflush(stdout);
        }
        std::fprintf(stderr,"# %s %lldx%lldx%lld alpha=%d\n",sh.tag,m,n,k,alpha);

        cudaFree(dA);cudaFree(dB);cudaFree(dCd);cudaFree(dCo);
        cudaFree(dAf);cudaFree(dBf);cudaFree(dCf);
        cudaFree(dSA);cudaFree(dSB);cudaFree(dEA);cudaFree(dEB);cudaFree(dCt);
        if(doCheck){cudaFree(dRhi);cudaFree(dRlo);}
    }
    cudaFree(dOut); CB(cublasDestroy(h));
    return 0;
}
