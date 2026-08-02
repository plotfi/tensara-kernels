// Solution for "matmul-swish-scaling": output[M,N] = swish(A[M,K] @ B[K,N]) * scale.
// Reuses the epilogue-fused GEMM body: plain producer, Compose<Swish, Scale> epilogue.
#include "../kernel-implementation/gemm-epilogue.cuh"

extern "C" void solution(const float* A, const float* B, float scale, float* output, size_t M, size_t N, size_t K) {
    auto ep = Compose<Swish, Scale>{ Swish{}, Scale{scale} };   // scale(swish(x))
    launch_gemm_epilogue<decltype(ep), /*B_T=*/false, /*HAS_BIAS=*/false>(
        A, B, nullptr, output, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), ep);
}
