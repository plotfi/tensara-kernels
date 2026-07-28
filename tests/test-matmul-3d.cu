// Correctness test for matmul-3d: batched A[n,m,k] * B[k,l] = C[n,m,l].
// For each batch a: C[a] = A[a] (m x k) * B (k x l).

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, float* C,
                         size_t n, size_t m, size_t k, size_t l);

int main() {
    test::seed();
    size_t n = 4, m = 64, k = 64, l = 32;

    float* h_a = new float[n * m * k];
    float* h_b = new float[k * l];
    float* h_c = new float[n * m * l];
    float* h_ref = new float[n * m * l];
    test::fill_random(h_a, n * m * k);
    test::fill_random(h_b, k * l);

    test::DBuf<float> d_a(n * m * k), d_b(k * l), d_c(n * m * l);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_c, n, m, k, l);
    test::check_cuda("matmul-3d");
    d_c.download(h_c);

    for (size_t a = 0; a < n; a++)
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < l; j++) {
                double acc = 0.0;
                for (size_t p = 0; p < k; p++)
                    acc += static_cast<double>(h_a[(a * m + i) * k + p]) * h_b[p * l + j];
                h_ref[(a * m + i) * l + j] = static_cast<float>(acc);
            }

    int bad = test::compare("matmul-3d", h_c, h_ref, n * m * l, 1e-3f, 1e-3f);
    int rc = test::report("matmul-3d", bad, n * m * l);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
