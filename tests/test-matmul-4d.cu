// Correctness test for matmul-4d: A[b,i,j,k] * B[k,l] = C[b,i,j,l].
// The leading three dims are flattened; note the solution() param order puts
// l before k: solution(A, B, C, b, i, j, l, k).

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, float* C,
                         size_t b, size_t i, size_t j, size_t l, size_t k);

int main() {
    test::seed();
    size_t b = 2, i = 4, j = 32, l = 32, k = 16;
    size_t R = b * i * j; // flattened rows

    float* h_a = new float[R * k];
    float* h_b = new float[k * l];
    float* h_c = new float[R * l];
    float* h_ref = new float[R * l];
    test::fill_random(h_a, R * k);
    test::fill_random(h_b, k * l);

    test::DBuf<float> d_a(R * k), d_b(k * l), d_c(R * l);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_c, b, i, j, l, k);
    test::check_cuda("matmul-4d");
    d_c.download(h_c);

    for (size_t r = 0; r < R; r++)
        for (size_t c = 0; c < l; c++) {
            double acc = 0.0;
            for (size_t p = 0; p < k; p++)
                acc += static_cast<double>(h_a[r * k + p]) * h_b[p * l + c];
            h_ref[r * l + c] = static_cast<float>(acc);
        }

    int bad = test::compare("matmul-4d", h_c, h_ref, R * l, 1e-3f, 1e-3f);
    int rc = test::report("matmul-4d", bad, R * l);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
