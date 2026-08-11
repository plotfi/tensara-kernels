// CS4803 (Spring 2010) Lab 3 — 2D convolution, SHARED-memory version.
// Ported from CS4803VGCD/lab3/lab3_shared/2Dconvolution_kernel.cu. Each block
// loads a 16x16 output tile plus a 2px halo into a 20x20 shared tile; only the
// inner 16x16 threads (indices 2..17) write output. 5x5 filter, size % 16 == 0.
#include <cuda_runtime.h>

#define KERNEL_SIZE 5
#define BLOCK_SIZE  16

typedef struct { unsigned int width, height, pitch; float* elements; } Matrix;

// --- original kernel, unchanged (dead debug comments removed) ---
__global__ void ConvolutionKernel(Matrix filterd, Matrix Nd, Matrix Pd) {
    __shared__ float Ms[KERNEL_SIZE][KERNEL_SIZE];
    __shared__ float Ns[20][20];

    int tx = threadIdx.x, ty = threadIdx.y;
    int by = blockIdx.y, bx = blockIdx.x;
    float result = 0;

    for (unsigned int m = 0; m < filterd.height; ++m)
        for (unsigned int n = 0; n < filterd.width; n++)
            Ms[m][n] = filterd.elements[m * filterd.width + n];

    if ((by == 0 && (ty == 1 || ty == 0)) ||
        (by == gridDim.y - 1 && (ty == 19 || ty == 18)) ||
        (bx == 0 && (tx == 1 || tx == 0)) ||
        (bx == gridDim.x - 1 && (tx == 19 || tx == 18))) {
        Ns[ty][tx] = 0;
    } else {
        Ns[ty][tx] = Nd.elements[(BLOCK_SIZE * Nd.width * by) + (bx * BLOCK_SIZE) +
                                 (ty - 2) * Nd.width + tx - 2];
    }
    __syncthreads();

    if (2 <= ty && 2 <= tx && ty <= 17 && tx <= 17) {
        for (unsigned int m = 0; m < 5; ++m)
            for (unsigned int n = 0; n < 5; n++)
                result += Ms[m][n] * Ns[ty + m - 2][tx + n - 2];
        Pd.elements[(BLOCK_SIZE * Pd.width * by) + (bx * BLOCK_SIZE) +
                    (ty - 2) * Pd.width + tx - 2] = result;
    }
}

extern "C" void solution(const float* filter, const float* N, float* P, size_t size) {
    Matrix Md{ KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, const_cast<float*>(filter) };
    Matrix Nd{ (unsigned)size, (unsigned)size, (unsigned)size, const_cast<float*>(N) };
    Matrix Pd{ (unsigned)size, (unsigned)size, (unsigned)size, P };
    dim3 threads(20, 20);
    dim3 blocks(size / BLOCK_SIZE, size / BLOCK_SIZE);
    ConvolutionKernel<<<blocks, threads>>>(Md, Nd, Pd);
}
