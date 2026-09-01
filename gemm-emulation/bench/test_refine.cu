/* Incremental refinement must be exact, not approximate.
 *
 * Splitting the work as [0,c) then [c,e) issues the same sequence of cuBLAS
 * calls, in the same order, as a single [0,e). So the refined result must be
 * BITWISE identical to computing e terms in one go. If that ever stops
 * holding, "spend c products, look at C, spend a few more" silently becomes
 * order-dependent, so it is worth pinning down.
 */

#include "mpemu/mpemu.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__global__ void fill(float* a, long long n, unsigned long long seed)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for(; i<n; i+=st){
        unsigned long long h=splitmix64(seed^(unsigned long long)i);
        double u=(double)(h>>11)*(1.0/9007199254740992.0);
        a[i]=(float)(u-0.5);
    }
}

static const long long N = 512;   /* small: this is an exactness test */

struct Ctx {
    cublasHandle_t h;
    float *dA,*dB,*dCa,*dCb;
    __nv_bfloat16 *dSAb,*dSBb;
    __half *dSAh,*dSBh;
    long long lds, str;
    float sA, sB;                 /* fp16 power-of-two scales */
    std::vector<float> ha, hb;
};

static bool identical(Ctx& c)
{
    CK(cudaMemcpy(c.ha.data(),c.dCa,(size_t)N*N*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.hb.data(),c.dCb,(size_t)N*N*4,cudaMemcpyDeviceToHost));
    return std::memcmp(c.ha.data(),c.hb.data(),(size_t)N*N*4)==0;
}

/* Compare monolithic [0,e) against [0,c) followed by [c,e). */
static int bf16Case(Ctx& c, int nsplits, int cut, int end)
{
    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,nsplits,0,end,0.0f,c.dCa,N));

    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,nsplits,0,cut,0.0f,c.dCb,N));
    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,nsplits,cut,end,1.0f,c.dCb,N));
    CK(cudaDeviceSynchronize());

    const bool ok=identical(c);
    std::printf("%s bf16  nsplits=%d  [0,%d) == [0,%d)+[%d,%d)\n",
                ok?"ok  ":"FAIL",nsplits,end,cut,cut,end);
    return ok?0:1;
}

static int fp16Case(Ctx& c, int nsplits, int cut, int end)
{
    MP(mpemuGemmMultiwordRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAh,c.lds,c.str,c.sA,c.dSBh,c.lds,c.str,c.sB,nsplits,0,end,0.0f,c.dCa,N));

    MP(mpemuGemmMultiwordRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAh,c.lds,c.str,c.sA,c.dSBh,c.lds,c.str,c.sB,nsplits,0,cut,0.0f,c.dCb,N));
    MP(mpemuGemmMultiwordRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAh,c.lds,c.str,c.sA,c.dSBh,c.lds,c.str,c.sB,nsplits,cut,end,1.0f,c.dCb,N));
    CK(cudaDeviceSynchronize());

    const bool ok=identical(c);
    std::printf("%s fp16  nsplits=%d  [0,%d) == [0,%d)+[%d,%d)\n",
                ok?"ok  ":"FAIL",nsplits,end,cut,cut,end);
    return ok?0:1;
}

int main()
{
    int bad=0;
    Ctx c;
    CB(cublasCreate(&c.h));
    c.lds=mpemuSplitLd(N); c.str=mpemuSplitStride(c.lds,N);
    c.ha.resize((size_t)N*N); c.hb.resize((size_t)N*N);

    CK(cudaMalloc(&c.dA,(size_t)N*N*4));  CK(cudaMalloc(&c.dB,(size_t)N*N*4));
    CK(cudaMalloc(&c.dCa,(size_t)N*N*4)); CK(cudaMalloc(&c.dCb,(size_t)N*N*4));
    CK(cudaMalloc(&c.dSAb,mpemuSplitBytes(N,N,MPEMU_MAX_SPLITS)));
    CK(cudaMalloc(&c.dSBb,mpemuSplitBytes(N,N,MPEMU_MAX_SPLITS)));
    CK(cudaMalloc(&c.dSAh,mpemuSplitBytes(N,N,MPEMU_MAX_SPLITS)));
    CK(cudaMalloc(&c.dSBh,mpemuSplitBytes(N,N,MPEMU_MAX_SPLITS)));

    fill<<<256,256>>>(c.dA,N*N,11ull);
    fill<<<256,256>>>(c.dB,N*N,22ull);

    std::printf("-- bf16: every split point of the 3-word schedule --\n");
    MP(mpemuSplitBF16(c.dA,N,N,N,3,c.dSAb,c.lds,c.str,0));
    MP(mpemuSplitBF16(c.dB,N,N,N,3,c.dSBb,c.lds,c.str,0));
    for (int cut=1; cut<9; ++cut) bad += bf16Case(c,3,cut,9);
    /* the schemes people actually stop at */
    bad += bf16Case(c,3,1,3);
    bad += bf16Case(c,3,3,6);
    bad += bf16Case(c,2,2,4);

    std::printf("-- fp16 multiword --\n");
    c.sA=1.0f; c.sB=1.0f;
    MP(mpemuAutoScaleFP16(c.dA,N,N,N,&c.sA,0));
    MP(mpemuAutoScaleFP16(c.dB,N,N,N,&c.sB,0));
    MP(mpemuSplitFP16(c.dA,N,N,N,3,c.sA,c.dSAh,c.lds,c.str,0));
    MP(mpemuSplitFP16(c.dB,N,N,N,3,c.sB,c.dSBh,c.lds,c.str,0));
    for (int cut=1; cut<6; ++cut) bad += fp16Case(c,3,cut,6);
    bad += fp16Case(c,2,1,3);          /* stop at double-fp16, refine within */
    bad += fp16Case(c,2,3,4);          /* extend past the i+j<=p+1 cut */

    /* MPEMU_ALL_MACS must mean "through the last term". */
    std::printf("-- MPEMU_ALL_MACS --\n");
    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,3,0,9,0.0f,c.dCa,N));
    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,3,0,4,0.0f,c.dCb,N));
    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,3,4,MPEMU_ALL_MACS,1.0f,c.dCb,N));
    CK(cudaDeviceSynchronize());
    { bool ok=identical(c); std::printf("%s [0,9) == [0,4)+[4,ALL)\n",ok?"ok  ":"FAIL");
      if(!ok) ++bad; }

    /* The old whole-budget driver must still agree with the range form. */
    MP(mpemuGemmEmulated(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,3,6,0.0f,c.dCa,N));
    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,3,0,6,0.0f,c.dCb,N));
    CK(cudaDeviceSynchronize());
    { bool ok=identical(c); std::printf("%s mpemuGemmEmulated(6) == Range(0,6)\n",ok?"ok  ":"FAIL");
      if(!ok) ++bad; }

    /* Empty range applies beta and nothing else. */
    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,3,0,6,0.0f,c.dCa,N));
    CK(cudaMemcpy(c.dCb,c.dCa,(size_t)N*N*4,cudaMemcpyDeviceToDevice));
    MP(mpemuGemmEmulatedRange(c.h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,1.0f,
        c.dSAb,c.lds,c.str,c.dSBb,c.lds,c.str,3,6,6,1.0f,c.dCb,N));
    CK(cudaDeviceSynchronize());
    { bool ok=identical(c); std::printf("%s empty range with beta=1 is a no-op\n",ok?"ok  ":"FAIL");
      if(!ok) ++bad; }

    std::printf("\n%s (%d failures)\n", bad?"FAILED":"ALL PASSED", bad);
    CB(cublasDestroy(c.h));
    return bad?1:0;
}
