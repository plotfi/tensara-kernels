#include <cuda_runtime.h>

#define BLOCK_SIZE 512

__global__ void _kernel(const float* X, float* Y, unsigned N) {
    /// threads ids inside each row
    #define tid threadIdx.x

    // one block per row, we launched grid as batch size
    const float *row = X + blockIdx.x * N;

    /// Collect x^2 sum for current thread
    float x_sum_sq = 0.0f;
    for (unsigned i = tid; i < N; i+= BLOCK_SIZE) {
        const float x = __ldca(row + i);
        x_sum_sq = fmaf(x, x, x_sum_sq);
    }

    /// Buffer for merging x^2 sums across threads
    __shared__ float merge_buf[BLOCK_SIZE];

    /// Tree reduction to merge x^2 sums across all threads block
    for (unsigned i = 1; i < BLOCK_SIZE; i *= 2) {
        merge_buf[tid] = x_sum_sq;
        __syncthreads();

        x_sum_sq += merge_buf[(tid + i) % BLOCK_SIZE];
        __syncthreads();
    }
    
    /// Calculate 1 / sqrt(mean(x^2) + episilon)
    #define e 1e-5f
    #define mean_x_sq __fdividef(x_sum_sq, N)
    const float rsqrt_mean_x_sq_e = rsqrtf(mean_x_sq + e);

    // 1:1 output block per row
    float *output = Y + blockIdx.x * N;
    for (unsigned i = tid; i < N; i+= BLOCK_SIZE) {
        const float x = __ldlu(row + i);
        output[i] = x * rsqrt_mean_x_sq_e;
    }
}

// Note: X, Y are device pointers
extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    unsigned grid = static_cast<unsigned>(B);
    unsigned n = static_cast<unsigned>(N);
    _kernel<<<grid, BLOCK_SIZE>>>(X, Y, n);
}
