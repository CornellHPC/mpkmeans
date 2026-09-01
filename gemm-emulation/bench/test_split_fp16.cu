/* Correctness checks for the multiword fp16 split.
 *
 * Unlike the 3-way bf16 split, the fp16 split is NOT expected to be exact:
 *  - p=2 covers 22 significand bits against fp32's 24, so the reconstruction
 *    residual should sit near u_low^2 = 2^-22 ~ 2.4e-07.
 *  - p=3 would cover 33 bits and could be exact, but the third word lands at
 *    ~2^-23 relative, which is SUBNORMAL in fp16 unless the operand is scaled
 *    up. This test pins down both behaviours.
 */

#include "mpemu/mpemu.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(x) do { cudaError_t e=(x); if (e!=cudaSuccess) {                    \
    std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1);} } while(0)

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


__global__ void fill(float* a, long long n, unsigned long long seed, float lo, float hi)
{
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for (; i<n; i+=st) a[i]=uniformFloat(seed,i,lo,hi);
}

/* Reconstruction error of sum(words)/scale vs the original, two ways.
 *
 * out[0] componentwise: max |sum-a| / |a|. Dominated by the smallest elements,
 *        whose trailing words fall below fp16's smallest subnormal -- so on a
 *        distribution containing near-zero entries this saturates and stops
 *        distinguishing schemes.
 * out[1] normwise: max |sum-a| / max|a|. This is what the multiword error
 *        analysis actually bounds, and it cleanly shows ~2^-11p. */
__global__ void reconK(const float* __restrict__ A, long long lda,
                       const __half* __restrict__ S, long long lds,
                       long long stride, int p, float scale,
                       long long rows, long long cols, float* out)
{
    long long total=rows*cols;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long st=(long long)gridDim.x*blockDim.x;
    for (; i<total; i+=st) {
        long long r=i%rows, c=i/rows;
        float a=A[c*lda+r];
        float s=0.0f;
        for (int j=0;j<p;++j) s+=__half2float(S[j*stride+c*lds+r]);
        s/=scale;
        float d=fabsf(s-a);
        float rel=(a!=0.0f)?d/fabsf(a):d;
        atomicMax((unsigned int*)&out[0],__float_as_uint(rel));
        atomicMax((unsigned int*)&out[1],__float_as_uint(d));
        atomicMax((unsigned int*)&out[2],__float_as_uint(fabsf(a)));
    }
}

static int run(long long rows,long long cols,int p,float lo,float hi,
               bool autoscale,double expectNormwise,const char* what)
{
    const long long lds=mpemuSplitLd(rows), str=mpemuSplitStride(lds,cols);
    float* dA; __half* dS; float* dM; unsigned long long* dC;
    CK(cudaMalloc(&dA,(size_t)rows*cols*4));
    CK(cudaMalloc(&dS,mpemuSplitBytes(rows,cols,p)));
    CK(cudaMalloc(&dM,3*4)); CK(cudaMemset(dM,0,3*4));
    CK(cudaMalloc(&dC,3*sizeof(unsigned long long)));
    CK(cudaMemset(dC,0,3*sizeof(unsigned long long)));

    fill<<<256,256>>>(dA,rows*cols,7ull,lo,hi);

    float scale=1.0f;
    if (autoscale) {
        mpemuStatus_t st=mpemuAutoScaleFP16(dA,rows,rows,cols,&scale,0);
        if (st!=MPEMU_STATUS_SUCCESS){ std::printf("FAIL %s autoscale\n",what); return 1; }
    }
    mpemuStatus_t st=mpemuSplitFP16(dA,rows,rows,cols,p,scale,dS,lds,str,0);
    if (st!=MPEMU_STATUS_SUCCESS){
        std::printf("FAIL %-30s split -> %s\n",what,mpemuStatusString(st)); return 1; }

    reconK<<<256,256>>>(dA,rows,dS,lds,str,p,scale,rows,cols,dM);
    CK(mpemuCheckSplitFP16(dA,rows,rows,cols,p,scale,dS,lds,str,dC,0)==MPEMU_STATUS_SUCCESS
       ? cudaSuccess : cudaErrorUnknown);
    CK(cudaDeviceSynchronize());

    float mm[3]; unsigned long long hc[3];
    CK(cudaMemcpy(mm,dM,3*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc,dC,sizeof hc,cudaMemcpyDeviceToHost));
    cudaFree(dA);cudaFree(dS);cudaFree(dM);cudaFree(dC);

    const double tot=(double)(hc[0]+hc[1]+hc[2]);
    const double normwise=(mm[2]>0.0f)?(double)mm[1]/(double)mm[2]:0.0;
    const bool ok=(normwise<=expectNormwise);
    std::printf("%s %-28s p=%d scale=%-9g normwise=%.3e (<=%.0e) "
                "componentwise=%.3e ovf=%llu subnorm=%.2f%%\n",
                ok?"ok  ":"FAIL", what, p, scale, normwise, expectNormwise,
                (double)mm[0], (unsigned long long)hc[0], 100.0*hc[1]/tot);
    return ok?0:1;
}

int main()
{
    int bad=0;

    /* --- Clean regime: one binade, away from zero, scaled so no word is
     * subnormal. Here the theory applies exactly: p words of fp16 carry 11p
     * significand bits, floored by fp32's own 24. --- */
    std::printf("-- single binade [0.25,0.5], scaled: theory applies cleanly --\n");
    bad += run(1024,1024,1,0.25f,0.5f,true, 1e-3,"p=1  (~2^-11 = 4.9e-4)");
    bad += run(1024,1024,2,0.25f,0.5f,true, 1e-6,"p=2  (~2^-22 = 2.4e-7)");
    bad += run(1024,1024,3,0.25f,0.5f,true, 1e-8,"p=3  (exact: 33 > 24 bits)");

    /* --- The paper's distribution. Contains near-zero entries whose trailing
     * words fall below fp16's smallest subnormal, so componentwise error
     * saturates; the normwise figure still behaves. --- */
    std::printf("-- U(-0.5,0.5], scaled --\n");
    bad += run(1024,1024,1,-0.5f,0.5f,true, 1e-3,"p=1");
    bad += run(1024,1024,2,-0.5f,0.5f,true, 1e-6,"p=2 (double-fp16)");
    bad += run(1024,1024,3,-0.5f,0.5f,true, 1e-7,"p=3");

    /* --- Unscaled: the exponent-range caveat, measured. The second word sits
     * near 2.4e-5, below fp16's smallest normal (6.1e-5). --- */
    std::printf("-- U(-0.5,0.5], UNSCALED (literal scheme) --\n");
    bad += run(1024,1024,2,-0.5f,0.5f,false,1e-4,"p=2 unscaled");
    bad += run(1024,1024,3,-0.5f,0.5f,false,1e-4,"p=3 unscaled");

    /* --- Small operands away from zero: scaling is what rescues them. --- */
    std::printf("-- small values [1e-3,2e-3] --\n");
    bad += run(1024,1024,2,1e-3f,2e-3f,false,1e-4,"p=2 unscaled");
    bad += run(1024,1024,2,1e-3f,2e-3f,true, 1e-6,"p=2 scaled");

    /* --- Shape edge cases: exercise the scalar (non-vectorised) path. --- */
    std::printf("-- shape edge cases --\n");
    bad += run(1023,777,2,0.25f,0.5f,true,1e-6,"p=2, row tail");
    bad += run(1000,333,2,0.25f,0.5f,true,1e-6,"p=2, odd shape");
    bad += run(7,3,2,0.25f,0.5f,true,1e-6,"p=2, tiny matrix");

    /* --- Recommended MAC counts: p(p+1)/2, the i+j <= p+1 rule. --- */
    const int want[4]={0,1,3,6};
    for (int p=1;p<=3;++p) {
        int got=mpemuMultiwordMacs(p);
        if (got!=want[p]) { std::printf("FAIL mpemuMultiwordMacs(%d)=%d want %d\n",
                                        p,got,want[p]); ++bad; }
    }
    std::printf("ok   mpemuMultiwordMacs: p=1->1, p=2->3, p=3->6\n");

    std::printf("\n%s (%d failures)\n", bad?"FAILED":"ALL PASSED", bad);
    return bad?1:0;
}
