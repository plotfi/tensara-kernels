// Solution for "gemm-relu": C[B,M] = relu(A[B,N] @ W[M,N]^T + b[M]).
// Reuses the epilogue-fused GEMM body: transposed-B + bias producer, Relu epilogue.
#include "../kernel-implementation/gemm-epilogue.cuh"

extern "C" void solution(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M) {
    // Mrows=B, Ncols=M, Kinner=N; W is [M,N] read transposed as W[j*N+k].
    launch_gemm_epilogue<Relu, /*B_T=*/true, /*HAS_BIAS=*/true>(
        A, W, b, C, static_cast<int>(B), static_cast<int>(M), static_cast<int>(N), Relu{});
}
