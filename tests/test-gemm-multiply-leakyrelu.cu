// Correctness test for gemm-multiply-leakyrelu.
// ASSUMED semantics: out = leaky_relu( (A @ B) * C, alpha ), elementwise, where
//   A[M,K], B[K,N], C[M,N], out[M,N], leaky_relu(x)=x>0?x:alpha*x.
// If your problem defines a different fusion order, adjust the reference below.

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, const float* C, float alpha,
                         float* output, size_t M, size_t N, size_t K);

int main() {
    test::seed();
    size_t M = 64, N = 64, K = 64;
    float alpha = 0.01f;

    float* h_a = new float[M * K];
    float* h_b = new float[K * N];
    float* h_c = new float[M * N];
    float* h_out = new float[M * N];
    float* h_ref = new float[M * N];
    test::fill_random(h_a, M * K);
    test::fill_random(h_b, K * N);
    test::fill_random(h_c, M * N);

    test::DBuf<float> d_a(M * K), d_b(K * N), d_c(M * N), d_out(M * N);
    d_a.upload(h_a); d_b.upload(h_b); d_c.upload(h_c);
    solution(d_a, d_b, d_c, alpha, d_out, M, N, K);
    test::check_cuda("gemm-multiply-leakyrelu");
    d_out.download(h_out);

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            double acc = 0.0;
            for (size_t p = 0; p < K; p++)
                acc += static_cast<double>(h_a[i * K + p]) * h_b[p * N + j];
            double v = acc * h_c[i * N + j];
            h_ref[i * N + j] = static_cast<float>(v > 0.0 ? v : alpha * v);
        }

    int bad = test::compare("gemm-multiply-leakyrelu", h_out, h_ref, M * N, 1e-3f, 1e-3f);
    int rc = test::report("gemm-multiply-leakyrelu", bad, M * N);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_out; delete[] h_ref;
    return rc;
}
