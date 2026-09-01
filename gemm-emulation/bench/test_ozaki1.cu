/* Correctness tests for Ozaki scheme I on INT8 tensor cores.
 *
 * The load-bearing claim is SLICE STABILITY: because each slice is an
 * independent bit-field of the shared-exponent fixed-point word, splitting
 * into 8 slices must reproduce the first 4 planes of a 4-slice split
 * byte-for-byte. That is what makes "spend a few products, then spend a few
 * more" cheap rather than a recomputation.
 */

#include "mpemu/mpemu.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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
/* uniform in (-0.5,0.5], optionally with a wide exponent spread */
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
            v*=exp(phi*(2.0*w-1.0)*3.0);
        }
        a[i]=v;
    }
}

/* Reconstruct |M| from its slices and report the worst normwise residual. */
__global__ void reconK(const double* __restrict__ M, long long ld,
                       const signed char* __restrict__ S, long long lds,
                       long long stride, int ns, int alpha,
                       const int* __restrict__ e,
                       long long rows, long long cols,
                       int leftOperand, double* out)
{
    long long total=rows*cols;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<total; i+=st){
        long long r=i%rows, c=i/rows;
        double v=M[c*ld+r];
        /* plane coordinates: left operand is stored transposed */
        long long pr = leftOperand ? c : r;
        long long pc = leftOperand ? r : c;
        int ex = e[pc];
        double acc=0.0;
        if(ex!=MPEMU_OZ1_ZERO_EXP){
            for(int p=0;p<ns;++p){
                double d=(double)S[p*stride+pc*lds+pr];
                acc += ldexp(d, ex-(p+1)*alpha);
            }
        }
        double diff=fabs(acc-v);
        atomicMax((unsigned long long*)&out[0], (unsigned long long)__double_as_longlong(diff));
        atomicMax((unsigned long long*)&out[1], (unsigned long long)__double_as_longlong(fabs(v)));
    }
}

/* double-double reference for A*B, ~106 bits. */
#define TS 16
__device__ __forceinline__ void ddAcc(double& sh, double& sl, double p, double e)
{
    double t=sh+p, bv=t-sh;
    sl += ((sh-(t-bv))+(p-bv)) + e;
    double s2=t+sl; sl=sl-(s2-t); sh=s2;
}
__global__ void ddRef(const double* __restrict__ A, long long lda,
                      const double* __restrict__ B, long long ldb,
                      double* __restrict__ Chi, double* __restrict__ Clo,
                      long long ldc, long long m, long long n, long long k)
{
    __shared__ double As[TS][TS+1], Bs[TS][TS+1];
    int tx=threadIdx.x, ty=threadIdx.y;
    long long row=(long long)blockIdx.x*TS+tx, col=(long long)blockIdx.y*TS+ty;
    double sh=0.0, sl=0.0;
    for(long long t=0;t<k;t+=TS){
        As[ty][tx]=(row<m && t+ty<k)?A[(t+ty)*lda+row]:0.0;
        Bs[tx][ty]=(col<n && t+tx<k)?B[col*ldb+(t+tx)]:0.0;
        __syncthreads();
#pragma unroll
        for(int i2=0;i2<TS;++i2){
            double a=As[i2][tx], b=Bs[i2][ty];
            double p=a*b, e=fma(a,b,-p);      /* exact product as a pair */
            ddAcc(sh,sl,p,e);
        }
        __syncthreads();
    }
    if(row<m&&col<n){ Chi[col*ldc+row]=sh; Clo[col*ldc+row]=sl; }
}
__global__ void errK(const double* __restrict__ C, long long ldc,
                     const double* __restrict__ Chi, const double* __restrict__ Clo,
                     long long ldr, long long rows, long long cols, double* out)
{
    __shared__ double sd[256], sr[256];
    double ad=0, ar=0;
    long long total=rows*cols;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<total; i+=st){
        long long r=i%rows,c=i/rows;
        double hi=Chi[c*ldr+r], lo=Clo[c*ldr+r];
        double d=((C[c*ldc+r]-hi)-lo), ref=hi+lo;
        ad+=d*d; ar+=ref*ref;
    }
    int t=threadIdx.x; sd[t]=ad; sr[t]=ar; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s){sd[t]+=sd[t+s];sr[t]+=sr[t+s];} __syncthreads(); }
    if(t==0){ atomicAdd(&out[0],sd[0]); atomicAdd(&out[1],sr[0]); }
}

static const long long N = 512;

int main()
{
    int bad=0;
    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));

    const int alpha = mpemuOz1BitsPerSlice(N);
    std::printf("-- parameters --\n");
    std::printf("ok   alpha = mpemuOz1BitsPerSlice(%lld) = %d  (expect 7 for k <= 2^17)\n", N, alpha);
    if (alpha != 7) { std::printf("FAIL expected alpha 7\n"); ++bad; }
    for (int s=1;s<=9;++s) {
        int want = s*(s+1)/2;
        if (mpemuOz1Macs(s)!=want){ std::printf("FAIL mpemuOz1Macs(%d)=%d want %d\n",s,mpemuOz1Macs(s),want); ++bad; }
    }
    std::printf("ok   mpemuOz1Macs: s(s+1)/2  (s=8 -> %d products)\n", mpemuOz1Macs(8));
    std::printf("ok   mpemuOz1MaxK(7) = %lld\n", mpemuOz1MaxK(7));

    const int SMAX=8;
    const long long lds=mpemuOz1SplitLd(N), stride=lds*N;
    double *dA,*dB,*dC1,*dC2,*dRhi,*dRlo,*dOut;
    signed char *dS4,*dS8,*dSB;
    int *dEA,*dEB,*dCt;
    CK(cudaMalloc(&dA,(size_t)N*N*8)); CK(cudaMalloc(&dB,(size_t)N*N*8));
    CK(cudaMalloc(&dC1,(size_t)N*N*8)); CK(cudaMalloc(&dC2,(size_t)N*N*8));
    CK(cudaMalloc(&dRhi,(size_t)N*N*8)); CK(cudaMalloc(&dRlo,(size_t)N*N*8));
    CK(cudaMalloc(&dOut,4*8));
    CK(cudaMalloc(&dS4,mpemuOz1SplitBytes(N,N,SMAX)));
    CK(cudaMalloc(&dS8,mpemuOz1SplitBytes(N,N,SMAX)));
    CK(cudaMalloc(&dSB,mpemuOz1SplitBytes(N,N,SMAX)));
    CK(cudaMalloc(&dEA,N*4)); CK(cudaMalloc(&dEB,N*4));
    CK(cudaMalloc(&dCt,(size_t)N*N*4));

    fill<<<256,256>>>(dA,N*N,11ull,0.0);
    fill<<<256,256>>>(dB,N*N,22ull,0.0);
    MP(mpemuOz1Exponents(dA,N,N,N,1,dEA,0));
    MP(mpemuOz1Exponents(dB,N,N,N,0,dEB,0));

    /* ---- THE claim: slices are stable when you ask for more of them ---- */
    std::printf("-- slice stability under extension --\n");
    CK(cudaMemset(dS4,0,mpemuOz1SplitBytes(N,N,SMAX)));
    CK(cudaMemset(dS8,0,mpemuOz1SplitBytes(N,N,SMAX)));
    MP(mpemuSplitInt8Ozaki1(dA,N,N,N,1,4,alpha,dEA,dS4,lds,stride,0));
    MP(mpemuSplitInt8Ozaki1(dA,N,N,N,1,8,alpha,dEA,dS8,lds,stride,0));
    CK(cudaDeviceSynchronize());
    {
        std::vector<signed char> a((size_t)stride*4), b((size_t)stride*4);
        CK(cudaMemcpy(a.data(),dS4,a.size(),cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(),dS8,b.size(),cudaMemcpyDeviceToHost));
        bool ok = std::memcmp(a.data(),b.data(),a.size())==0;
        std::printf("%s split(s=8) planes 0..3 are byte-identical to split(s=4)\n", ok?"ok  ":"FAIL");
        if(!ok) ++bad;
    }

    /* ---- slice values must fit INT8 ---- */
    {
        std::vector<signed char> v((size_t)stride*8);
        CK(cudaMemcpy(v.data(),dS8,v.size(),cudaMemcpyDeviceToHost));
        int lim=(1<<alpha)-1, worst=0;
        for(size_t i=0;i<v.size();++i){ int m=v[i]<0?-v[i]:v[i]; if(m>worst) worst=m; }
        bool ok = worst<=lim;
        std::printf("%s slice magnitudes <= 2^alpha-1 (max %d, limit %d)\n", ok?"ok  ":"FAIL",worst,lim);
        if(!ok) ++bad;
    }

    /* ---- reconstruction: s slices should recover ~s*alpha bits ---- */
    std::printf("-- reconstruction accuracy (normwise) --\n");
    for (int s : {2,4,6,8}) {
        MP(mpemuSplitInt8Ozaki1(dA,N,N,N,1,s,alpha,dEA,dS8,lds,stride,0));
        CK(cudaMemset(dOut,0,2*8));
        reconK<<<256,256>>>(dA,N,dS8,lds,stride,s,alpha,dEA,N,N,1,dOut);
        CK(cudaDeviceSynchronize());
        double o[2]; CK(cudaMemcpy(o,dOut,2*8,cudaMemcpyDeviceToHost));
        double rel=o[0]/o[1];
        double expect = ldexp(1.0, -(s*alpha)) * 4.0;      /* a few ulp of slack */
        bool ok = rel <= expect || (s*alpha >= 53 && rel <= 1e-15);
        std::printf("%s s=%d (%2d bits): normwise residual %.3e  (bound %.1e)\n",
                    ok?"ok  ":"FAIL", s, s*alpha, rel, expect);
        if(!ok) ++bad;
    }

    /* ---- GEMM accuracy against a double-double reference ---- */
    std::printf("-- GEMM vs cublasDgemm, against a ~106-bit reference --\n");
    {
        dim3 blk(TS,TS), grd((unsigned)((N+TS-1)/TS),(unsigned)((N+TS-1)/TS));
        ddRef<<<grd,blk>>>(dA,N,dB,N,dRhi,dRlo,N,N,N,N);
        CK(cudaDeviceSynchronize());
    }
    auto relerr=[&](const double* C)->double{
        CK(cudaMemset(dOut,0,2*8));
        errK<<<1024,256>>>(C,N,dRhi,dRlo,N,N,N,dOut);
        CK(cudaDeviceSynchronize());
        double o[2]; CK(cudaMemcpy(o,dOut,2*8,cudaMemcpyDeviceToHost));
        return o[1]>0?sqrt(o[0]/o[1]):0.0;
    };
    { const double one=1.0,zero=0.0;
      CB(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,(int)N,(int)N,(int)N,&one,dA,(int)N,dB,(int)N,&zero,dC1,(int)N));
      CK(cudaDeviceSynchronize());
      std::printf("ok   cublasDgemm                 rel %.3e\n", relerr(dC1)); }

    MP(mpemuSplitInt8Ozaki1(dB,N,N,N,0,SMAX,alpha,dEB,dSB,lds,stride,0));
    for (int s : {2,3,4,5,6,7,8}) {
        MP(mpemuSplitInt8Ozaki1(dA,N,N,N,1,s,alpha,dEA,dS8,lds,stride,0));
        MP(mpemuGemmOzaki1(h,N,N,N,1.0,dS8,lds,stride,dEA,dSB,lds,stride,dEB,
                           s,alpha,0.0,dC2,N,dCt,N,0));
        CK(cudaDeviceSynchronize());
        std::printf("ok   Ozaki I s=%d (%2d GEMMs)      rel %.3e\n", s, mpemuOz1Macs(s), relerr(dC2));
    }

    /* ---- refinement: bitwise-exact at bin boundaries ---- */
    std::printf("-- refinement (bin boundaries = mpemuOz1Macs(s)) --\n");
    MP(mpemuSplitInt8Ozaki1(dA,N,N,N,1,SMAX,alpha,dEA,dS8,lds,stride,0));
    std::vector<double> h1((size_t)N*N), h2((size_t)N*N);
    for (int s=2;s<=SMAX;++s) {
        const int cut=mpemuOz1Macs(s-1), end=mpemuOz1Macs(s);
        MP(mpemuGemmOzaki1Range(h,N,N,N,1.0,dS8,lds,stride,dEA,dSB,lds,stride,dEB,
                                SMAX,alpha,0,end,0.0,dC1,N,dCt,N,0));
        MP(mpemuGemmOzaki1Range(h,N,N,N,1.0,dS8,lds,stride,dEA,dSB,lds,stride,dEB,
                                SMAX,alpha,0,cut,0.0,dC2,N,dCt,N,0));
        MP(mpemuGemmOzaki1Range(h,N,N,N,1.0,dS8,lds,stride,dEA,dSB,lds,stride,dEB,
                                SMAX,alpha,cut,end,1.0,dC2,N,dCt,N,0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h1.data(),dC1,(size_t)N*N*8,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(),dC2,(size_t)N*N*8,cudaMemcpyDeviceToHost));
        bool ok=std::memcmp(h1.data(),h2.data(),(size_t)N*N*8)==0;
        std::printf("%s [0,%d) == [0,%d)+[%d,%d)   (s=%d from s=%d)\n",
                    ok?"ok  ":"FAIL", end, cut, cut, end, s, s-1);
        if(!ok) ++bad;
    }

    std::printf("\n%s (%d failures)\n", bad?"FAILED":"ALL PASSED", bad);
    CB(cublasDestroy(h));
    return bad?1:0;
}
