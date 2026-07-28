// Correctness test for max-normalize: y = x / max(|x|), per row (inf-norm).

#include "test_utils.cuh"
extern "C" void solution(const float* X, float* Y, size_t B, size_t D);

int main() {
    test::seed();
    size_t B = 8, D = 64; // D must be a multiple of 4

    float* h_x   = new float[B * D];
    float* h_out = new float[B * D];
    float* h_ref = new float[B * D];
    test::fill_random(h_x, B * D);

    test::DBuf<float> d_x(B * D), d_y(B * D);
    d_x.upload(h_x);
    solution(d_x, d_y, B, D);
    test::check_cuda("max-normalize");
    d_y.download(h_out);

    for (size_t r = 0; r < B; r++) {
        float m = 0.0f;
        for (size_t j = 0; j < D; j++) m = fmaxf(m, fabsf(h_x[r * D + j]));
        for (size_t j = 0; j < D; j++) h_ref[r * D + j] = h_x[r * D + j] / m;
    }

    int bad = test::compare("max-normalize", h_out, h_ref, B * D, 1e-2f, 1e-3f);
    int rc = test::report("max-normalize", bad, B * D);

    delete[] h_x; delete[] h_out; delete[] h_ref;
    return rc;
}
