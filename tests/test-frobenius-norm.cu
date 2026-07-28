// Correctness test for frobenius-norm normalization: Y = X / ||X||_F,
// where ||X||_F = sqrt(sum_i X[i]^2). Input and output both have `size` elements.

#include "test_utils.cuh"
extern "C" void solution(const float* X, float* Y, size_t size);

int main() {
    test::seed();
    size_t size = 4096;

    float* h_x = new float[size];
    float* h_y = new float[size];
    float* h_ref = new float[size];
    test::fill_random(h_x, size);

    test::DBuf<float> d_x(size), d_y(size);
    d_x.upload(h_x);
    solution(d_x, d_y, size);
    test::check_cuda("frobenius-norm");
    d_y.download(h_y);

    double acc = 0.0;
    for (size_t i = 0; i < size; i++) acc += (double)h_x[i] * h_x[i];
    float norm = sqrtf(static_cast<float>(acc));
    for (size_t i = 0; i < size; i++) h_ref[i] = h_x[i] / norm;

    int bad = test::compare("frobenius-norm", h_y, h_ref, size, 1e-3f, 1e-5f);
    int rc = test::report("frobenius-norm", bad, size);
    delete[] h_x; delete[] h_y; delete[] h_ref;
    return rc;
}
