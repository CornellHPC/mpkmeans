/* The concatenated-k variant rests on one index identity: sliding the stacked
 * reversed right operand's base by (nsplits-1-b)*k pairs A's block i with B's
 * block b-i for every i simultaneously. If that is off by one the result is
 * silently a different (wrong) set of products, so check it against the
 * per-product Range drivers, which are independently tested.
 */

#include "mpemu/mpemu.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
__global__ void fill(float* a,long long n,unsigned long long seed)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st){
        unsigned long long h=sm64(seed^(unsigned long long)i);
        a[i]=(float)((double)(h>>11)*(1.0/9007199254740992.0)-0.5);
    }
}
__global__ void diffK(const float* X,const float* Y,long long n,double* out)
{
    __shared__ double sd[256],sr[256];
    double ad=0,ar=0;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st){ double d=(double)X[i]-(double)Y[i]; ad+=d*d; ar+=(double)Y[i]*(double)Y[i]; }
    int t=threadIdx.x; sd[t]=ad; sr[t]=ar; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s){sd[t]+=sd[t+s];sr[t]+=sr[t+s];} __syncthreads(); }
    if(t==0){ atomicAdd(&out[0],sd[0]); atomicAdd(&out[1],sr[0]); }
}

int main()
{
    int bad=0;
    cublasHandle_t h; CB(cublasCreate(&h));
    CB(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH));

    /* deliberately non-square and not a nice multiple */
    const long long m=384, n=192, k=100;

    float *dA,*dB,*dC1,*dC2; double* dOut;
    CK(cudaMalloc(&dA,(size_t)m*k*4)); CK(cudaMalloc(&dB,(size_t)k*n*4));
    CK(cudaMalloc(&dC1,(size_t)m*n*4)); CK(cudaMalloc(&dC2,(size_t)m*n*4));
    CK(cudaMalloc(&dOut,2*sizeof(double)));
    fill<<<64,256>>>(dA,m*k,11ull);
    fill<<<64,256>>>(dB,k*n,22ull);

    const long long lda=mpemuSplitLd(m), strA=lda*k;
    const long long ldb=mpemuSplitLd(k), strB=ldb*n;

    auto reldiff=[&](){
        CK(cudaMemset(dOut,0,2*sizeof(double)));
        diffK<<<64,256>>>(dC1,dC2,m*n,dOut);
        CK(cudaDeviceSynchronize()); double o[2];
        CK(cudaMemcpy(o,dOut,2*sizeof(double),cudaMemcpyDeviceToHost));
        return o[1]>0?sqrt(o[0]/o[1]):0.0; };

    std::printf("-- bf16: concat vs per-product Range --\n");
    for (int S=1; S<=3; ++S) {
        __nv_bfloat16 *sA,*sBn,*sBs;
        const long long lds=mpemuStackedLd(k,S);
        CK(cudaMalloc(&sA,(size_t)strA*S*2));
        CK(cudaMalloc(&sBn,(size_t)strB*S*2));
        CK(cudaMalloc(&sBs,mpemuStackedBytes(k,n,S,2)));
        MP(mpemuSplitBF16(dA,m,m,k,S,sA,lda,strA,0));
        MP(mpemuSplitBF16(dB,k,k,n,S,sBn,ldb,strB,0));
        MP(mpemuSplitBF16Stacked(dB,k,k,n,S,sBs,lds,0));

        for (int nb=1; nb<=S; ++nb) {
            const int prod=nb*(nb+1)/2;
            MP(mpemuGemmEmulatedConcat(h,m,n,k,1.0f,sA,lda,sBs,lds,S,nb,0.0f,dC1,m));
            MP(mpemuGemmEmulatedRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,
                                      sA,lda,strA,sBn,ldb,strB,S,0,prod,0.0f,dC2,m));
            CK(cudaDeviceSynchronize());
            const double d=reldiff();
            const bool ok = d < 1e-6;
            std::printf("%s splits=%d bins=%d (%d products)  rel diff %.3e\n",
                        ok?"ok  ":"FAIL",S,nb,prod,d);
            if(!ok) ++bad;
        }
        cudaFree(sA);cudaFree(sBn);cudaFree(sBs);
    }

    std::printf("-- multiword fp16: concat vs per-product Range --\n");
    for (int S=1; S<=3; ++S) {
        __half *sA,*sBn,*sBs;
        const long long lds=mpemuStackedLd(k,S);
        CK(cudaMalloc(&sA,(size_t)strA*S*2));
        CK(cudaMalloc(&sBn,(size_t)strB*S*2));
        CK(cudaMalloc(&sBs,mpemuStackedBytes(k,n,S,2)));
        float scA=1.0f,scB=1.0f;
        MP(mpemuAutoScaleFP16(dA,m,m,k,&scA,0));
        MP(mpemuAutoScaleFP16(dB,k,k,n,&scB,0));
        MP(mpemuSplitFP16(dA,m,m,k,S,scA,sA,lda,strA,0));
        MP(mpemuSplitFP16(dB,k,k,n,S,scB,sBn,ldb,strB,0));
        MP(mpemuSplitFP16Stacked(dB,k,k,n,S,scB,sBs,lds,0));

        for (int nb=1; nb<=S; ++nb) {
            const int prod=nb*(nb+1)/2;
            MP(mpemuGemmMultiwordConcat(h,m,n,k,1.0f,scA,scB,sA,lda,sBs,lds,S,nb,0.0f,dC1,m));
            MP(mpemuGemmMultiwordRange(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,1.0f,
                                       sA,lda,strA,scA,sBn,ldb,strB,scB,S,0,prod,0.0f,dC2,m));
            CK(cudaDeviceSynchronize());
            const double d=reldiff();
            const bool ok = d < 1e-6;
            std::printf("%s splits=%d bins=%d (%d products)  rel diff %.3e\n",
                        ok?"ok  ":"FAIL",S,nb,prod,d);
            if(!ok) ++bad;
        }
        cudaFree(sA);cudaFree(sBn);cudaFree(sBs);
    }

    std::printf("\n%s (%d failures)\n", bad?"FAILED":"ALL PASSED", bad);
    CB(cublasDestroy(h));
    return bad?1:0;
}
