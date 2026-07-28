// Correctness test for gemm-relu: a linear layer with ReLU.
// A[B,N], weight W[M,N], bias b[M] -> C[B,M] = relu(A @ W^T + b).
// C[i,j] = relu( sum_n A[i,n]*W[j,n] + b[j] ).

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* W, const float* b, float* C,
                         size_t B, size_t N, size_t M);

int main() {
    test::seed();
    size_t B = 8, N = 64, M = 32;

    float* h_a = new float[B * N];
    float* h_w = new float[M * N];
    float* h_b = new float[M];
    float* h_c = new float[B * M];
    float* h_ref = new float[B * M];
    test::fill_random(h_a, B * N);
    test::fill_random(h_w, M * N);
    test::fill_random(h_b, M);

    test::DBuf<float> d_a(B * N), d_w(M * N), d_b(M), d_c(B * M);
    d_a.upload(h_a); d_w.upload(h_w); d_b.upload(h_b);
    solution(d_a, d_w, d_b, d_c, B, N, M);
    test::check_cuda("gemm-relu");
    d_c.download(h_c);

    for (size_t i = 0; i < B; i++)
        for (size_t j = 0; j < M; j++) {
            double acc = h_b[j];
            for (size_t n = 0; n < N; n++)
                acc += static_cast<double>(h_a[i * N + n]) * h_w[j * N + n];
            h_ref[i * M + j] = acc > 0.0 ? static_cast<float>(acc) : 0.0f;
        }

    int bad = test::compare("gemm-relu", h_c, h_ref, B * M, 1e-3f, 1e-3f);
    int rc = test::report("gemm-relu", bad, B * M);
    delete[] h_a; delete[] h_w; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
