#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

// Templated on ACTIVATION function pointer and BLOCK_SIZE so each activation
// gets its own specialized kernel with a tuned launch config. The activation
// is inlined at every call via __forceinline__ on the free function.
//
// __restrict__ tells the compiler A and C do not alias, unlocking better
// instruction scheduling (loads and stores can be reordered freely, all loads
// can be issued up front to hide memory latency).
//
// n is int (32 bit) instead of size_t (64 bit). 32 bit index math lowers to
// simpler SASS: address calcs and loop increments avoid the 64 bit multiply
// and carry chain that size_t forces. Meaningful savings on tight loops.
template <float (*ACTIVATION)(float, float), int BLOCK_SIZE>
__global__ void activation_kernelx8(const float* __restrict__ A,
                                    float* __restrict__ C,
                                    int n,
                                    float alpha) {
    // Each thread handles 8 elements. B200 tensor cores and load/store units
    // can move 256 bits at a time (two float4s back to back), so 8 floats per
    // thread lets the compiler keep two 128 bit loads in flight simultaneously,
    // maxing out memory-level parallelism per thread.
    int base = (threadIdx.x + blockIdx.x * blockDim.x) * 8;

    // Scalar tail for the last partial group of 8. Uses the same ACTIVATION
    // so the tail matches the vectorized path exactly.
    if (base + 7 >= n) {
        for (int j = base; j < n; ++j) {
            C[j] = ACTIVATION(A[j], alpha);
        }
        return;
    }

    // Both float4 loads issued back to back. The compiler keeps both
    // LDG.E.128 instructions in flight without waiting on either, giving the
    // memory subsystem two outstanding requests per thread to hide HBM latency.
    float4 x0 = *reinterpret_cast<const float4*>(A + base);
    float4 x1 = *reinterpret_cast<const float4*>(A + base + 4);

    // 8 independent activation evaluations per thread. With __forceinline__ on
    // each activation, these lower to 8 straight-line SFU or ALU sequences
    // with no register spills and full ILP exposure.
    x0.x = ACTIVATION(x0.x, alpha); x0.y = ACTIVATION(x0.y, alpha);
    x0.z = ACTIVATION(x0.z, alpha); x0.w = ACTIVATION(x0.w, alpha);
    x1.x = ACTIVATION(x1.x, alpha); x1.y = ACTIVATION(x1.y, alpha);
    x1.z = ACTIVATION(x1.z, alpha); x1.w = ACTIVATION(x1.w, alpha);

    // Two 128 bit stores, mirroring the load pattern. STG.E.128 x 2.
    *reinterpret_cast<float4*>(C + base    ) = x0;
    *reinterpret_cast<float4*>(C + base + 4) = x1;
}

// Host-side launcher. Marked __forceinline__ so it collapses into solution()
// at the call site. No perf impact worth measuring, just cleaner codegen.
template <float (*ACTIVATION)(float, float), int BLOCK_SIZE>
__forceinline__ void multi_solution(const float* input, float* output, size_t n, size_t m,
                                    float alpha) {
    // Cast to int for the same 32 bit arithmetic reason as the kernel.
    int N = static_cast<int>(n * m);
    if (N == 0) return;

    // Each thread does 8 elements, so we need ceil(N/8) threads total.
    int threads_needed = (N + 7) / 8;
    const int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;

    activation_kernelx8<ACTIVATION, BLOCK_SIZE>
        <<<grid, BLOCK_SIZE>>>(input, output, N, alpha);
}

__device__ __forceinline__ float fast_tanh(float x) {
    float e2x = __expf(2.0f * x);
    return __fdividef(e2x - 1.0f, e2x + 1.0f);
}

// All activations share the same (float x, float alpha) signature. Stateless
// activations just ignore alpha. This uniform signature lets the kernel take
// any of them as a template argument without a functor wrapper.
__device__ __forceinline__ float relu(float x, float)             { return x > 0.0f ? x : 0.0f; }
__device__ __forceinline__ float leaky_relu(float x, float alpha) { return x > 0.0f ? x : x * alpha; }
__device__ __forceinline__ float elu(float x, float alpha)        { return x > 0.0f ? x : alpha * (__expf(x) - 1.0f); }
__device__ __forceinline__ float sigmoid(float x, float)          { return __fdividef(1.0f, 1.0f + __expf(-x)); }
__device__ __forceinline__ float swish(float x, float)            { return x * __fdividef(1.0f, 1.0f + __expf(-x)); }
__device__ __forceinline__ float tanh_act(float x, float)         { return fast_tanh(x); }
__device__ __forceinline__ float gelu(float x, float) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoef = 0.044715f;
    return 0.5f * x * (1.0f + fast_tanh(kSqrt2OverPi * (x + kCoef * x * x * x)));
}
__device__ __forceinline__ float selu(float x, float) {
    return 1.0507f * (fmaxf(0.0f, x) + fminf(0.0f, 1.67326f * (__expf(x) - 1.0f)));
}
__device__ __forceinline__ float softplus(float x, float)         { return __logf(1.0f + __expf(x)); }
__device__ __forceinline__ float hard_sigmoid(float x, float) {
    return x <= -3.0f ? 0.0f : (x >= 3.0f ? 1.0f : __fdividef(x + 3.0f, 6.0f));
}
