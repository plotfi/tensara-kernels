// Correctness test for matmul-sigmoid-sum.
// ASSUMED semantics: output[0] = sum over all elements of sigmoid(A @ B), where
//   A[M,K], B[K,N]. (Output buffer has size 1.)
// If your problem reduces along a specific dim instead, adjust the reference.

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, float* output,
                         size_t M, size_t N, size_t K);

int main() {
    test::seed();
    size_t M = 64, N = 64, K = 64;

    float* h_a = new float[M * K];
    float* h_b = new float[K * N];
    float h_o = 0.0f, h_ref = 0.0f;
    test::fill_random(h_a, M * K);
    test::fill_random(h_b, K * N);

    test::DBuf<float> d_a(M * K), d_b(K * N), d_o(1);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_o, M, N, K);
    test::check_cuda("matmul-sigmoid-sum");
    d_o.download(&h_o);

    double sum = 0.0;
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            double acc = 0.0;
            for (size_t p = 0; p < K; p++)
                acc += static_cast<double>(h_a[i * K + p]) * h_b[p * N + j];
            sum += 1.0 / (1.0 + exp(-acc));
        }
    h_ref = static_cast<float>(sum);

    // Sum over 4096 elements: use a relative tolerance.
    int bad = test::compare("matmul-sigmoid-sum", &h_o, &h_ref, 1, 1e-3f, 1e-2f);
    int rc = test::report("matmul-sigmoid-sum", bad, 1);
    delete[] h_a; delete[] h_b;
    return rc;
}
