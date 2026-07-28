// Correctness test for matmul-swish-scaling.
// ASSUMED semantics: output = swish(A @ B) * scale, elementwise, where
//   A[M,K], B[K,N], swish(x) = x * sigmoid(x).

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, float scale, float* output,
                         size_t M, size_t N, size_t K);

int main() {
    test::seed();
    size_t M = 64, N = 64, K = 64;
    float scale = 1.0f;

    float* h_a = new float[M * K];
    float* h_b = new float[K * N];
    float* h_o = new float[M * N];
    float* h_ref = new float[M * N];
    test::fill_random(h_a, M * K);
    test::fill_random(h_b, K * N);

    test::DBuf<float> d_a(M * K), d_b(K * N), d_o(M * N);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, scale, d_o, M, N, K);
    test::check_cuda("matmul-swish-scaling");
    d_o.download(h_o);

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            double acc = 0.0;
            for (size_t p = 0; p < K; p++)
                acc += static_cast<double>(h_a[i * K + p]) * h_b[p * N + j];
            double sw = acc / (1.0 + exp(-acc));
            h_ref[i * N + j] = static_cast<float>(sw * scale);
        }

    int bad = test::compare("matmul-swish-scaling", h_o, h_ref, M * N, 1e-3f, 1e-3f);
    int rc = test::report("matmul-swish-scaling", bad, M * N);
    delete[] h_a; delete[] h_b; delete[] h_o; delete[] h_ref;
    return rc;
}
