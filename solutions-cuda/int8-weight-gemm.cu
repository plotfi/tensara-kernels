// Solution for "int8-weight-gemm": weight-only int8 group-quantized linear layer
//   C[M,N] = A[M,K] @ dequant(Wq[N,K], scale[N,K/G])^T
// The dequant is fused into the matmul load (a prologue) -- the fp32 weight is
// never materialized. See kernel-implementation/gemm-prologue.cuh.
#include "../kernel-implementation/gemm-prologue.cuh"

extern "C" void solution(const float* A, const int8_t* Wq, const float* scale,
                         float* C, size_t M, size_t N, size_t K, int group_size) {
    if (M == 0 || N == 0) return;
    launch_gemm_prologue(A, Wq, scale, C,
                         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                         group_size, DequantInt8{});
}
