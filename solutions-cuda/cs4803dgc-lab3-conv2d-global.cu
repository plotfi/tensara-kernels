// CS4803 (Spring 2010) Lab 3 — 2D convolution, GLOBAL-memory version.
// Ported from CS4803VGCD/lab3/lab3_global/2Dconvolution_kernel.cu. 5x5 filter,
// same-size zero-padded output. Reads the input straight from global memory.
#include <cuda_runtime.h>

#define KERNEL_SIZE 5
#define BLOCK_SIZE  16

typedef struct { unsigned int width, height, pitch; float* elements; } Matrix;

// --- original kernel, unchanged ---
__global__ void ConvolutionKernel(Matrix filterd, Matrix Nd, Matrix Pd) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;
    float result = 0;
    unsigned int mbegin = (row < 2) ? 2 - row : 0;
    unsigned int mend   = (row > (Nd.height - 3)) ? Nd.height - row + 2 : 5;
    unsigned int nbegin = (col < 2) ? 2 - col : 0;
    unsigned int nend   = (col > (Nd.width - 3)) ? (Nd.width - col) + 2 : 5;
    for (unsigned int m = mbegin; m < mend; ++m)
        for (unsigned int n = nbegin; n < nend; n++)
            result += filterd.elements[m * 5 + n] *
                      Nd.elements[Nd.width * (row + m - 2) + (col + n - 2)];
    if (col < Pd.width && row < Pd.height)
        Pd.elements[row * Pd.width + col] = result;
}

// Playground contract: filter is 5x5, image N and output P are size x size.
extern "C" void solution(const float* filter, const float* N, float* P, size_t size) {
    Matrix Md{ KERNEL_SIZE, KERNEL_SIZE, KERNEL_SIZE, const_cast<float*>(filter) };
    Matrix Nd{ (unsigned)size, (unsigned)size, (unsigned)size, const_cast<float*>(N) };
    Matrix Pd{ (unsigned)size, (unsigned)size, (unsigned)size, P };
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(size / BLOCK_SIZE, size / BLOCK_SIZE);
    ConvolutionKernel<<<blocks, threads>>>(Md, Nd, Pd);
}
