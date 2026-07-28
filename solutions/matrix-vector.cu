#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 128
#define tid threadIdx.x

__global__ void matmul(const float* A, const float* B, float* C, int K) {
    /// one block per row, we launched grid as batch size
    const float *row_a = A + blockIdx.x * K;

    /// Accumulator
    float acc = 0.0f;

    const int K2 = K >> 1;   // number of float2s

    const float2* row_a2 = reinterpret_cast<const float2*>(row_a);
    const float2* B2     = reinterpret_cast<const float2*>(B);

    // i is a float2 index now: start at tid, stride by BLOCK_SIZE
    for (int i = tid; i < K2; i += BLOCK_SIZE) {
        float2 r = row_a2[i];
        float2 b = B2[i];
        acc = fmaf(r.x, b.x, acc);
        acc = fmaf(r.y, b.y, acc);
    }

    /// Buffer for merging x^2 sums across threads
    __shared__ float merge_buf[BLOCK_SIZE];

    /// Tree reduction to merge x^2 sums across all threads block
    for (int i = 1; i < BLOCK_SIZE; i *= 2) {
        merge_buf[tid] = acc;
        __syncthreads();

        acc += merge_buf[(tid + i) % BLOCK_SIZE];
        __syncthreads();
    }

    /// Only one thread needs to write the result,
    /// but the values are identical so we can leave the race condition.
    C[blockIdx.x] = acc;
}

// Note: input_a, input_b, output_c are device pointers
extern "C" void solution(const float* input_a, const float* input_b, float* output_c,
                         size_t m, size_t k) {
    int grid = static_cast<int>(m);
    int K = static_cast<int>(k);
    matmul<<<grid, BLOCK_SIZE>>>(input_a, input_b, output_c, K);    
}
