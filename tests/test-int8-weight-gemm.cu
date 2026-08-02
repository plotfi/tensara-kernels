// Correctness test for int8-weight-gemm (weight-only int8 group-quantized GEMM).
// Reference: C[m,n] = sum_k A[m,k] * (Wq[n,k] * scale[n, k/G]), symmetric int8
// (zero-point 0), group size G along K.

#include "test_utils.cuh"
#include <cstdint>
extern "C" void solution(const float* A, const int8_t* Wq, const float* scale,
                         float* C, size_t M, size_t N, size_t K, int group_size);

int main() {
    test::seed();
    int M = 64, N = 64, K = 256, G = 64;
    int NG = K / G;                       // groups per output row

    float*  h_a     = new float[M * K];
    int8_t* h_wq    = new int8_t[N * K];
    float*  h_scale = new float[N * NG];
    float*  h_c     = new float[M * N];
    float*  h_ref   = new float[M * N];

    test::fill_random(h_a, M * K);
    for (int i = 0; i < N * K; i++)  h_wq[i]    = static_cast<int8_t>((rand() & 0xff) - 128);
    for (int i = 0; i < N * NG; i++) h_scale[i] = 0.002f + (rand() % 100) * 0.001f;   // positive scales

    test::DBuf<float>  d_a(M * K);
    test::DBuf<int8_t> d_wq(N * K);
    test::DBuf<float>  d_scale(N * NG);
    test::DBuf<float>  d_c(M * N);
    d_a.upload(h_a); d_wq.upload(h_wq); d_scale.upload(h_scale);

    solution(d_a, d_wq, d_scale, d_c, M, N, K, G);
    test::check_cuda("int8-weight-gemm");
    d_c.download(h_c);

    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                double w = static_cast<double>(h_wq[n * K + k]) * h_scale[n * NG + k / G];
                acc += static_cast<double>(h_a[m * K + k]) * w;
            }
            h_ref[m * N + n] = static_cast<float>(acc);
        }

    int bad = test::compare("int8-weight-gemm", h_c, h_ref, M * N, 1e-3f, 1e-3f);
    int rc = test::report("int8-weight-gemm", bad, M * N);
    delete[] h_a; delete[] h_wq; delete[] h_scale; delete[] h_c; delete[] h_ref;
    return rc;
}
