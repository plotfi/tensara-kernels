// Correctness test for matrix-scalar: out = in * scalar (n x n).

#include "test_utils.cuh"
extern "C" void solution(const float* input_matrix, float scalar, float* output_matrix, size_t n);

int main() {
    test::seed();
    size_t n = 64;
    float scalar = 2.5f;

    float* h_in  = new float[n * n];
    float* h_out = new float[n * n];
    float* h_ref = new float[n * n];
    test::fill_random(h_in, n * n);

    test::DBuf<float> d_in(n * n), d_out(n * n);
    d_in.upload(h_in);
    solution(d_in, scalar, d_out, n);
    test::check_cuda("matrix-scalar");
    d_out.download(h_out);

    for (size_t i = 0; i < n * n; i++) h_ref[i] = h_in[i] * scalar;

    int bad = test::compare("matrix-scalar", h_out, h_ref, n * n, 1e-5f, 1e-6f);
    int rc = test::report("matrix-scalar", bad, n * n);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
