// Correctness test for layer-norm over the (F,D1,D2) features of each sample b.
// For each b: mean/var over the C=F*D1*D2 elements, then
//   Y = (X - mean) / sqrt(var + eps) * gamma[c] + beta[c]
// gamma/beta are indexed by the within-sample position c. eps=1e-5, population var.

#include "test_utils.cuh"
extern "C" void solution(const float* X, const float* gamma, const float* beta, float* Y,
                         size_t B, size_t F, size_t D1, size_t D2);

int main() {
    test::seed();
    size_t B = 2, F = 4, D1 = 8, D2 = 8;
    size_t C = F * D1 * D2;
    size_t total = B * C;
    const float eps = 1e-5f;

    float* h_x = new float[total];
    float* h_g = new float[C];
    float* h_b = new float[C];
    float* h_y = new float[total];
    float* h_ref = new float[total];
    test::fill_random(h_x, total);
    test::fill_random(h_g, C);
    test::fill_random(h_b, C);

    test::DBuf<float> d_x(total), d_g(C), d_bt(C), d_y(total);
    d_x.upload(h_x); d_g.upload(h_g); d_bt.upload(h_b);
    solution(d_x, d_g, d_bt, d_y, B, F, D1, D2);
    test::check_cuda("layer-norm");
    d_y.download(h_y);

    for (size_t b = 0; b < B; b++) {
        const float* x = h_x + b * C;
        double mean = 0.0;
        for (size_t c = 0; c < C; c++) mean += x[c];
        mean /= C;
        double var = 0.0;
        for (size_t c = 0; c < C; c++) { double dd = x[c] - mean; var += dd * dd; }
        var /= C;
        float inv = 1.0f / sqrtf(static_cast<float>(var) + eps);
        for (size_t c = 0; c < C; c++)
            h_ref[b * C + c] = (static_cast<float>(x[c] - mean)) * inv * h_g[c] + h_b[c];
    }

    int bad = test::compare("layer-norm", h_y, h_ref, total, 1e-3f, 1e-4f);
    int rc = test::report("layer-norm", bad, total);
    delete[] h_x; delete[] h_g; delete[] h_b; delete[] h_y; delete[] h_ref;
    return rc;
}
