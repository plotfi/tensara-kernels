// Correctness test for batch-norm (per-channel, using batch statistics, no affine).
// Layout X[B,F,D1,D2]. For each channel f: mean/var over the B*D1*D2 elements, then
//   Y = (X - mean_f) / sqrt(var_f + eps)
// eps=1e-5, population variance.

#include "test_utils.cuh"
extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2);

int main() {
    test::seed();
    size_t B = 2, F = 4, D1 = 8, D2 = 8;
    size_t HW = D1 * D2;
    size_t total = B * F * HW;
    const float eps = 1e-5f;

    float* h_x = new float[total];
    float* h_y = new float[total];
    float* h_ref = new float[total];
    test::fill_random(h_x, total);

    test::DBuf<float> d_x(total), d_y(total);
    d_x.upload(h_x);
    solution(d_x, d_y, B, F, D1, D2);
    test::check_cuda("batch-norm");
    d_y.download(h_y);

    auto idx = [&](size_t b, size_t f, size_t s) { return (b * F + f) * HW + s; };
    size_t cnt = B * HW;
    for (size_t f = 0; f < F; f++) {
        double mean = 0.0;
        for (size_t b = 0; b < B; b++) for (size_t s = 0; s < HW; s++) mean += h_x[idx(b, f, s)];
        mean /= cnt;
        double var = 0.0;
        for (size_t b = 0; b < B; b++) for (size_t s = 0; s < HW; s++) {
            double dd = h_x[idx(b, f, s)] - mean; var += dd * dd;
        }
        var /= cnt;
        float inv = 1.0f / sqrtf(static_cast<float>(var) + eps);
        for (size_t b = 0; b < B; b++) for (size_t s = 0; s < HW; s++)
            h_ref[idx(b, f, s)] = static_cast<float>(h_x[idx(b, f, s)] - mean) * inv;
    }

    int bad = test::compare("batch-norm", h_y, h_ref, total, 1e-3f, 1e-4f);
    int rc = test::report("batch-norm", bad, total);
    delete[] h_x; delete[] h_y; delete[] h_ref;
    return rc;
}
