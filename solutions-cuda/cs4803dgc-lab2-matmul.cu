// CS4803 (Spring 2010) Lab 2 — tiled/shared-memory matrix multiply, C = A * B.
// Ported verbatim from spring2010gradschool/CS4803VGCD/lab2/matrixmul_kernel.cu:
// BLOCK_SIZE x BLOCK_SIZE shared tiles, grid over blocks, bounds-checked (handles
// arbitrary sizes, unlike lab1's single-block naive version).
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

// --- original kernel, unchanged (typeof(*M.elements) -> float) ---
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
    __shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;

    float result = 0;
    for (int block = 0; block < M.width; block += BLOCK_SIZE) {
        int xoffset = block + tx;
        int yoffset = block + ty;
        Ms[ty][tx] = (xoffset < M.width && row < M.height) ? M.elements[row * M.width + xoffset] : 0;
        Ns[ty][tx] = (yoffset < N.height && col < N.width) ? N.elements[yoffset * N.width + col] : 0;
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
            result += Ms[ty][k] * Ns[k][tx];
        __syncthreads();
    }
    if (col < P.width && row < P.height)
        P.elements[row * P.width + col] = result;
}

// Playground contract: square n x n matrices, C = A * B.
extern "C" void solution(const float* A, const float* B, float* C, size_t n) {
    Matrix M{ (unsigned)n, (unsigned)n, (unsigned)n, const_cast<float*>(A) };
    Matrix N{ (unsigned)n, (unsigned)n, (unsigned)n, const_cast<float*>(B) };
    Matrix P{ (unsigned)n, (unsigned)n, (unsigned)n, C };
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    MatrixMulKernel<<<blocks, threads>>>(M, N, P);
}
