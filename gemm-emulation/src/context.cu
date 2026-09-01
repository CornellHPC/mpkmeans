/* One growable device arena, carved into every scheme's workspace.
 *
 * The split/GEMM routines themselves never allocate. This exists so a caller
 * can hoist every temporary out of a loop: for small operands a
 * cudaMalloc/cudaFree pair costs more than the GEMM it brackets, so a
 * steady-state iteration should perform no CUDA allocation at all.
 */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

#include <cstdint>
#include <cstdlib>

namespace {

constexpr size_t kAlign = 256;   /* keeps every view comfortably aligned */

inline size_t alignUp(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

/* Bump-carve a view out of the arena. */
struct Carve {
    char*  base;
    size_t cap;
    size_t off;
    bool   ok;

    explicit Carve(void* p, size_t c) : base((char*)p), cap(c), off(0), ok(true) {}
    void* take(size_t bytes)
    {
        const size_t start = alignUp(off);
        if (start + bytes > cap) { ok = false; return nullptr; }
        off = start + bytes;
        return base + start;
    }
};

}  /* namespace */

struct mpemuContext_s {
    void*  arena;
    size_t capacity;
    int    allocs;
};

extern "C" mpemuStatus_t mpemuContextCreate(mpemuContext_t* ctx)
{
    if (!ctx) return MPEMU_STATUS_INVALID_VALUE;
    mpemuContext_t c = (mpemuContext_t)std::calloc(1, sizeof(mpemuContext_s));
    if (!c) return MPEMU_STATUS_CUDA_ERROR;
    c->arena = nullptr; c->capacity = 0; c->allocs = 0;
    *ctx = c;
    return MPEMU_STATUS_SUCCESS;
}

extern "C" mpemuStatus_t mpemuContextDestroy(mpemuContext_t ctx)
{
    if (!ctx) return MPEMU_STATUS_SUCCESS;
    if (ctx->arena) cudaFree(ctx->arena);
    std::free(ctx);
    return MPEMU_STATUS_SUCCESS;
}

extern "C" mpemuStatus_t mpemuContextReserve(mpemuContext_t ctx, size_t bytes)
{
    if (!ctx) return MPEMU_STATUS_INVALID_VALUE;
    if (bytes <= ctx->capacity) return MPEMU_STATUS_SUCCESS;   /* already fits */

    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return MPEMU_STATUS_CUDA_ERROR;
    if (ctx->arena) cudaFree(ctx->arena);
    ctx->arena = p;
    ctx->capacity = bytes;
    ++ctx->allocs;
    return MPEMU_STATUS_SUCCESS;
}

extern "C" size_t mpemuContextCapacity(mpemuContext_t ctx)
{ return ctx ? ctx->capacity : 0; }

extern "C" void* mpemuContextBase(mpemuContext_t ctx)
{ return ctx ? ctx->arena : nullptr; }

extern "C" int mpemuContextAllocCount(mpemuContext_t ctx)
{ return ctx ? ctx->allocs : 0; }

/* ------------------------------------------------------------------ */
/* Sizing                                                              */
/* ------------------------------------------------------------------ */

extern "C" size_t mpemuBf16WorkspaceBytes(int64_t m, int64_t n, int64_t k, int nsplits)
{
    if (nsplits < 1 || m < 0 || n < 0 || k < 0) return 0;
    const size_t a = (size_t)mpemuSplitLd(m) * (size_t)k * nsplits * sizeof(__nv_bfloat16);
    const size_t b = (size_t)mpemuSplitLd(k) * (size_t)n * nsplits * sizeof(__nv_bfloat16);
    return alignUp(a) + alignUp(b);
}

extern "C" size_t mpemuFp16WorkspaceBytes(int64_t m, int64_t n, int64_t k, int nsplits)
{
    /* fp16 and bf16 are both 2 bytes and share the split layout */
    return mpemuBf16WorkspaceBytes(m, n, k, nsplits);
}

extern "C" size_t mpemuOz1WorkspaceBytes(int64_t m, int64_t n, int64_t k, int nslices)
{
    if (nslices < 1 || m < 0 || n < 0 || k < 0) return 0;
    const size_t a  = (size_t)mpemuOz1SplitLd(k) * (size_t)m * nslices;  /* A^T planes */
    const size_t b  = (size_t)mpemuOz1SplitLd(k) * (size_t)n * nslices;
    const size_t ea = (size_t)m * sizeof(int);
    const size_t eb = (size_t)n * sizeof(int);
    const size_t ct = (size_t)m * (size_t)n * sizeof(int);
    return alignUp(a) + alignUp(b) + alignUp(ea) + alignUp(eb) + alignUp(ct);
}

/* ------------------------------------------------------------------ */
/* Carving                                                             */
/* ------------------------------------------------------------------ */

extern "C" mpemuStatus_t mpemuBf16WorkspaceInit(mpemuContext_t ctx,
                                                int64_t m, int64_t n, int64_t k,
                                                int nsplits,
                                                mpemuBf16Workspace_t* ws)
{
    if (!ctx || !ws || nsplits < 1) return MPEMU_STATUS_INVALID_VALUE;
    Carve c(ctx->arena, ctx->capacity);
    ws->ldA = mpemuSplitLd(m); ws->strideA = ws->ldA * k;
    ws->ldB = mpemuSplitLd(k); ws->strideB = ws->ldB * n;
    ws->sA = (__nv_bfloat16*)c.take((size_t)ws->strideA * nsplits * sizeof(__nv_bfloat16));
    ws->sB = (__nv_bfloat16*)c.take((size_t)ws->strideB * nsplits * sizeof(__nv_bfloat16));
    ws->nsplits = nsplits;
    return c.ok ? MPEMU_STATUS_SUCCESS : MPEMU_STATUS_INVALID_VALUE;
}

extern "C" mpemuStatus_t mpemuFp16WorkspaceInit(mpemuContext_t ctx,
                                                int64_t m, int64_t n, int64_t k,
                                                int nsplits,
                                                mpemuFp16Workspace_t* ws)
{
    if (!ctx || !ws || nsplits < 1) return MPEMU_STATUS_INVALID_VALUE;
    Carve c(ctx->arena, ctx->capacity);
    ws->ldA = mpemuSplitLd(m); ws->strideA = ws->ldA * k;
    ws->ldB = mpemuSplitLd(k); ws->strideB = ws->ldB * n;
    ws->sA = (__half*)c.take((size_t)ws->strideA * nsplits * sizeof(__half));
    ws->sB = (__half*)c.take((size_t)ws->strideB * nsplits * sizeof(__half));
    ws->nsplits = nsplits;
    return c.ok ? MPEMU_STATUS_SUCCESS : MPEMU_STATUS_INVALID_VALUE;
}

extern "C" mpemuStatus_t mpemuOz1WorkspaceInit(mpemuContext_t ctx,
                                               int64_t m, int64_t n, int64_t k,
                                               int nslices,
                                               mpemuOz1Workspace_t* ws)
{
    if (!ctx || !ws || nslices < 1) return MPEMU_STATUS_INVALID_VALUE;
    Carve c(ctx->arena, ctx->capacity);
    ws->ldA = mpemuOz1SplitLd(k); ws->strideA = ws->ldA * m;   /* A^T: k x m */
    ws->ldB = mpemuOz1SplitLd(k); ws->strideB = ws->ldB * n;
    ws->sA   = (signed char*)c.take((size_t)ws->strideA * nslices);
    ws->sB   = (signed char*)c.take((size_t)ws->strideB * nslices);
    ws->expA = (int*)c.take((size_t)m * sizeof(int));
    ws->expB = (int*)c.take((size_t)n * sizeof(int));
    ws->ct   = (int*)c.take((size_t)m * (size_t)n * sizeof(int));
    ws->ldct = m;
    ws->nslices   = nslices;
    ws->alphaBits = mpemuOz1BitsPerSlice(k);
    return c.ok ? MPEMU_STATUS_SUCCESS : MPEMU_STATUS_INVALID_VALUE;
}
