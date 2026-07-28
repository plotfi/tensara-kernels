// Correctness test for vector-addition: C[i] = A[i] + B[i].

#include "test_utils.cuh"
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n);

int main() {
    test::seed();

    // Use a non-multiple-of-16 size to exercise the scalar tail path too.
    size_t n = 1024 + 7;

    float* h_a   = new float[n];
    float* h_b   = new float[n];
    float* h_out = new float[n];
    float* h_ref = new float[n];
    test::fill_random(h_a, n);
    test::fill_random(h_b, n);

    test::DBuf<float> d_a(n), d_b(n), d_out(n);
    d_a.upload(h_a);
    d_b.upload(h_b);

    solution(d_a, d_b, d_out, n);
    test::check_cuda("vector-addition");

    d_out.download(h_out);
    for (size_t i = 0; i < n; i++)
        h_ref[i] = h_a[i] + h_b[i];

    int bad = test::compare("vector-addition", h_out, h_ref, n, 1e-5f, 1e-6f);
    int rc = test::report("vector-addition", bad, n);

    delete[] h_a; delete[] h_b; delete[] h_out; delete[] h_ref;
    return rc;
}
