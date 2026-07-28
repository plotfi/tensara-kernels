// Correctness test for symmetric-matmul: C = A * B where A, B are symmetric.
// The test constructs symmetric inputs so the result is unambiguous whether or
// not the kernel exploits symmetry, then compares against a full matmul.

#include "test_utils.cuh"
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n);

static void symmetrize(float* M, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = i + 1; j < n; j++)
            M[j * n + i] = M[i * n + j];
}

int main() {
    test::seed();
    size_t n = 64;

    float* h_a = new float[n * n];
    float* h_b = new float[n * n];
    float* h_c = new float[n * n];
    float* h_ref = new float[n * n];
    test::fill_random(h_a, n * n);
    test::fill_random(h_b, n * n);
    symmetrize(h_a, n);
    symmetrize(h_b, n);

    test::DBuf<float> d_a(n * n), d_b(n * n), d_c(n * n);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_c, n);
    test::check_cuda("symmetric-matmul");
    d_c.download(h_c);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (size_t p = 0; p < n; p++)
                acc += static_cast<double>(h_a[i * n + p]) * h_b[p * n + j];
            h_ref[i * n + j] = static_cast<float>(acc);
        }

    int bad = test::compare("symmetric-matmul", h_c, h_ref, n * n, 1e-3f, 1e-3f);
    int rc = test::report("symmetric-matmul", bad, n * n);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
