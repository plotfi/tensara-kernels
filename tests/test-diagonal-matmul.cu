// Correctness test for diagonal-matmul: C = diag(a) * B.
// C[i,j] = a[i] * B[i,j], with a of length n and B of shape n x m.

#include "test_utils.cuh"
extern "C" void solution(const float* diagonal_a, const float* input_b, float* output_c,
                         size_t n, size_t m);

int main() {
    test::seed();
    size_t n = 64, m = 64;

    float* h_a = new float[n];
    float* h_b = new float[n * m];
    float* h_c = new float[n * m];
    float* h_ref = new float[n * m];
    test::fill_random(h_a, n);
    test::fill_random(h_b, n * m);

    test::DBuf<float> d_a(n), d_b(n * m), d_c(n * m);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_c, n, m);
    test::check_cuda("diagonal-matmul");
    d_c.download(h_c);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            h_ref[i * m + j] = h_a[i] * h_b[i * m + j];

    int bad = test::compare("diagonal-matmul", h_c, h_ref, n * m, 1e-5f, 1e-6f);
    int rc = test::report("diagonal-matmul", bad, n * m);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
