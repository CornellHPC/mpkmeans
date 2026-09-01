#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

/* 8 BF16 elements = 16 bytes: keeps every split column 16-byte aligned for the
 * vectorised split kernel and meets the alignment cuBLAS prefers for
 * BF16 tensor-core GEMMs. */
extern "C" int64_t mpemuSplitLd(int64_t rows)
{
    return (rows + 7) / 8 * 8;
}

extern "C" int64_t mpemuSplitStride(int64_t lds, int64_t cols)
{
    return lds * cols;
}

extern "C" size_t mpemuSplitBytes(int64_t rows, int64_t cols, int nsplits)
{
    if (nsplits < 1 || nsplits > MPEMU_MAX_SPLITS) return 0;
    const int64_t lds = mpemuSplitLd(rows);
    return (size_t)mpemuSplitStride(lds, cols) * (size_t)nsplits
         * sizeof(__nv_bfloat16);
}

extern "C" const char* mpemuStatusString(mpemuStatus_t s)
{
    switch (s) {
        case MPEMU_STATUS_SUCCESS:       return "MPEMU_STATUS_SUCCESS";
        case MPEMU_STATUS_INVALID_VALUE: return "MPEMU_STATUS_INVALID_VALUE";
        case MPEMU_STATUS_CUDA_ERROR:    return "MPEMU_STATUS_CUDA_ERROR";
        case MPEMU_STATUS_CUBLAS_ERROR:  return "MPEMU_STATUS_CUBLAS_ERROR";
        default:                         return "MPEMU_STATUS_UNKNOWN";
    }
}
