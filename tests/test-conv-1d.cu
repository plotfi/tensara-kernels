// Correctness test for conv-1d: "same" 1D convolution with a centered kernel.
// C[i] = sum_j A[i + j - r] * B[j], r = (K-1)/2, out-of-range taps skipped.
// The kernel accumulates into C, which relies on C starting zeroed (DBuf does).

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K);

int main() {
    test::seed();

    size_t N = 1024;
    size_t K = 5;

    float* h_a   = new float[N];
    float* h_b   = new float[K];
    float* h_out = new float[N];
    float* h_ref = new float[N];
    test::fill_random(h_a, N);
    test::fill_random(h_b, K);

    test::DBuf<float> d_a(N), d_b(K), d_out(N);
    d_a.upload(h_a);
    d_b.upload(h_b);

    solution(d_a, d_b, d_out, N, K);
    test::check_cuda("conv-1d");

    d_out.download(h_out);
    unsigned r = (K - 1) / 2;
    for (size_t i = 0; i < N; i++) {
        double acc = 0.0;
        for (size_t j = 0; j < K; j++) {
            long kk = static_cast<long>(i) + static_cast<long>(j) - r;
            if (kk < 0 || kk >= static_cast<long>(N)) continue;
            acc += static_cast<double>(h_a[kk]) * h_b[j];
        }
        h_ref[i] = static_cast<float>(acc);
    }

    int bad = test::compare("conv-1d", h_out, h_ref, N, 1e-4f, 1e-5f);
    int rc = test::report("conv-1d", bad, N);

    delete[] h_a; delete[] h_b; delete[] h_out; delete[] h_ref;
    return rc;
}
