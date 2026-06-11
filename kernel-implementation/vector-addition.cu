#include <cuda_runtime.h>

__global__ void add_kernel_x16(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int n) {
    int base = (threadIdx.x + blockIdx.x * blockDim.x) * 16;

    int base1 = base + 4;
    int base2 = base + 8;
    int base3 = base + 12;

    if (base + 15 >= n) {
        // tail
        for (int j = base; j < n; ++j) {
            C[j] = A[j] + B[j];
        }
        return;
    }

    float4 a0 = __ldg(reinterpret_cast<const float4*>(A + base));
    float4 b0 = __ldg(reinterpret_cast<const float4*>(B + base));
    float4 a1 = __ldg(reinterpret_cast<const float4*>(A + base1));
    float4 b1 = __ldg(reinterpret_cast<const float4*>(B + base1));
    float4 a2 = __ldg(reinterpret_cast<const float4*>(A + base2));
    float4 b2 = __ldg(reinterpret_cast<const float4*>(B + base2));
    float4 a3 = __ldg(reinterpret_cast<const float4*>(A + base3));
    float4 b3 = __ldg(reinterpret_cast<const float4*>(B + base3));

    a0.x += b0.x; a0.y += b0.y; a0.z += b0.z; a0.w += b0.w;
    a1.x += b1.x; a1.y += b1.y; a1.z += b1.z; a1.w += b1.w;
    a2.x += b2.x; a2.y += b2.y; a2.z += b2.z; a2.w += b2.w;
    a3.x += b3.x; a3.y += b3.y; a3.z += b3.z; a3.w += b3.w;

    *reinterpret_cast<float4*>(C + base ) = a0;
    *reinterpret_cast<float4*>(C + base1) = a1;
    *reinterpret_cast<float4*>(C + base2) = a2;
    *reinterpret_cast<float4*>(C + base3) = a3;
}

extern "C" void solution(const float* d_input1, const float* d_input2,
                         float* d_output, size_t n) {
    int N = static_cast<int>(n);
    if (N == 0) return;

    const int BLOCK_SIZE = 256;
    int threads_needed = (N + 15) / 16;
    int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;

    add_kernel_x16<<<grid, BLOCK_SIZE>>>(d_input1, d_input2, d_output, N);
}
