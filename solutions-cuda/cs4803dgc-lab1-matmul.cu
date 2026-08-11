// CS4803 (Spring 2010) Lab 1 — naive matrix multiply, C = A * B.
// Ported verbatim from spring2010gradschool/CS4803VGCD/lab1/matrixmul_kernel.cu:
// one thread per output element, a SINGLE block (uses threadIdx only), so it only
// handles matrices up to one block (square n <= 32). Kept as-is as a reference.
#include <cuda_runtime.h>

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

// --- original kernel, unchanged ---
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {
    int MRow = threadIdx.y;
    int NCol = threadIdx.x;
    float mulValue = 0;
    for (int i = 0; i < M.width; i++) {
        float Melement = M.elements[MRow * M.width + i];
        float Nelement = N.elements[i * N.width + NCol];
        mulValue += Melement * Nelement;
    }
    P.elements[MRow * N.width + NCol] = mulValue;
}

// Playground contract: square n x n matrices, C = A * B. Single block (n <= 32).
extern "C" void solution(const float* A, const float* B, float* C, size_t n) {
    Matrix M{ (unsigned)n, (unsigned)n, (unsigned)n, const_cast<float*>(A) };
    Matrix N{ (unsigned)n, (unsigned)n, (unsigned)n, const_cast<float*>(B) };
    Matrix P{ (unsigned)n, (unsigned)n, (unsigned)n, C };
    dim3 threads((unsigned)n, (unsigned)n);   // one block, one thread per element
    MatrixMulKernel<<<1, threads>>>(M, N, P);
}
