// Correctness test for matrix-vector: C[i] = sum_k A[i,k] * B[k].
// The kernel reads the row in float2 pairs, so K must be even.

#include "test_utils.cuh"
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k);

int main() {
    test::seed();

    size_t m = 64, k = 64;

    float* h_a   = new float[m * k];
    float* h_b   = new float[k];
    float* h_out = new float[m];
    float* h_ref = new float[m];
    test::fill_random(h_a, m * k);
    test::fill_random(h_b, k);

    test::DBuf<float> d_a(m * k), d_b(k), d_out(m);
    d_a.upload(h_a);
    d_b.upload(h_b);

    solution(d_a, d_b, d_out, m, k);
    test::check_cuda("matrix-vector");

    d_out.download(h_out);
    for (size_t i = 0; i < m; i++) {
        double acc = 0.0;
        for (size_t j = 0; j < k; j++)
            acc += static_cast<double>(h_a[i * k + j]) * h_b[j];
        h_ref[i] = static_cast<float>(acc);
    }

    int bad = test::compare("matrix-vector", h_out, h_ref, m, 1e-3f, 1e-4f);
    int rc = test::report("matrix-vector", bad, m);

    delete[] h_a; delete[] h_b; delete[] h_out; delete[] h_ref;
    return rc;
}
