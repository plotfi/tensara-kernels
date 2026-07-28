// Correctness test for matrix-multiplication: C[m,n] = sum_k A[m,k] * B[k,n].
// Build:  make -C tests build/bin/test-matrix-multiplication.exe SOLUTION=your.cu

#include "test_utils.cuh"
extern "C" void solution(const float* input_a, const float* input_b, float* output_c,
                         size_t m, size_t n, size_t k);

int main() {
    test::seed();
    size_t m = 64, n = 64, k = 64;

    float* h_a = new float[m * k];
    float* h_b = new float[k * n];
    float* h_c = new float[m * n];
    float* h_ref = new float[m * n];
    test::fill_random(h_a, m * k);
    test::fill_random(h_b, k * n);

    test::DBuf<float> d_a(m * k), d_b(k * n), d_c(m * n);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_c, m, n, k);
    test::check_cuda("matrix-multiplication");
    d_c.download(h_c);

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (size_t p = 0; p < k; p++)
                acc += static_cast<double>(h_a[i * k + p]) * h_b[p * n + j];
            h_ref[i * n + j] = static_cast<float>(acc);
        }

    int bad = test::compare("matrix-multiplication", h_c, h_ref, m * n, 1e-3f, 1e-3f);
    int rc = test::report("matrix-multiplication", bad, m * n);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
