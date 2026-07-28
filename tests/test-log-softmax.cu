// Correctness test for log-softmax: y = x - log(sum(exp(x))), per row.

#include "test_utils.cuh"
extern "C" void solution(const float* input, float* output, size_t M, size_t N);

int main() {
    test::seed();
    size_t M = 64, N = 64; // N must be a multiple of 4

    float* h_x   = new float[M * N];
    float* h_out = new float[M * N];
    float* h_ref = new float[M * N];
    test::fill_random(h_x, M * N);

    test::DBuf<float> d_x(M * N), d_y(M * N);
    d_x.upload(h_x);
    solution(d_x, d_y, M, N);
    test::check_cuda("log-softmax");
    d_y.download(h_out);

    for (size_t r = 0; r < M; r++) {
        double acc = 0.0;
        for (size_t j = 0; j < N; j++) acc += exp(static_cast<double>(h_x[r * N + j]));
        float logsum = logf(static_cast<float>(acc));
        for (size_t j = 0; j < N; j++) h_ref[r * N + j] = h_x[r * N + j] - logsum;
    }

    int bad = test::compare("log-softmax", h_out, h_ref, M * N, 1e-2f, 1e-3f);
    int rc = test::report("log-softmax", bad, M * N);

    delete[] h_x; delete[] h_out; delete[] h_ref;
    return rc;
}
