/* Synthetic data and centroid initialisation. */
#include "mpk_internal.cuh"

#include <random>
#include <vector>
#include <algorithm>

extern "C" void mpkMakeBlobs(int n, int d, int k, float std, float center_box,
                             unsigned seed, float* hP, int* hLabels) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ubox(-center_box, center_box);
    std::normal_distribution<float> gauss(0.f, std);

    std::vector<float> centers((size_t)k * d);
    for (size_t t = 0; t < centers.size(); ++t) centers[t] = ubox(rng);

    for (int i = 0; i < n; ++i) {
        const int c = (int)((long long)i * k / n);   /* balanced clusters */
        if (hLabels) hLabels[i] = c;
        if (!hP) continue;
        for (int t = 0; t < d; ++t)
            hP[(size_t)i * d + t] = centers[(size_t)c * d + t] + gauss(rng);
    }
}

extern "C" mpkStatus mpkInitRandomPoints(const float* dP, int n, int d, int k,
                                         unsigned seed, float* dC) {
    if (k > n) return MPK_ERR_INVALID;
    std::mt19937 rng(seed ^ 0x9e3779b9u);

    /* partial Fisher-Yates over the index range, without materialising n ints
     * when n is large: reservoir-free selection via a hash set is overkill for
     * the sizes here, so just shuffle a vector. */
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> pick(i, n - 1);
        std::swap(idx[i], idx[pick(rng)]);
    }
    std::sort(idx.begin(), idx.begin() + k);

    for (int j = 0; j < k; ++j) {
        MPK_CUDA(cudaMemcpy(dC + (size_t)j * d, dP + (size_t)idx[j] * d,
                            (size_t)d * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    return MPK_OK;
}
