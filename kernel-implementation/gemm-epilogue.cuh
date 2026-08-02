#pragma once
#include <cuda_runtime.h>
#include "scalar-ops.cuh"   // epilogue functors (Relu/Swish/Scale/Identity/Compose/...)

// Epilogue-fused GEMM: one kernel body computes C = A @ B (+bias), then applies
// an arbitrary elementwise *epilogue* to each output element before the store.
// The epilogue is a functor with a __forceinline__ operator()(float)->float, so
// nvcc inlines the whole chain into the store site -- no call overhead, the
// value stays in a register, and the extra kernel launch + global round-trip
// that an unfused "matmul then activation" would pay are both eliminated. That
// is the fusion win.
//
// The epilogue ops (Relu, Swish, Scale, Identity, Compose, ...) come from
// scalar-ops.cuh -- the same functors the activation kernels use, so the math
// is defined once for both.
//
// The producer itself is the same for every fused variant; only the epilogue
// changes. B layout (normal vs transposed) and bias are compile-time flags so
// they cost nothing at runtime and still let one body serve A@B and A@W^T+bias.

// ---- fused GEMM kernel -----------------------------------------------------
//
// C[i,j] = ep( (HAS_BIAS ? bias[j] : 0) + sum_k A[i,k] * B(k,j) )
//   B_T == false: B is [Kinner, Ncols], B(k,j) = B[k*Ncols + j]      (A @ B)
//   B_T == true : B is [Ncols, Kinner], B(k,j) = B[j*Kinner + k]     (A @ B^T)
// One thread per output element (correctness-first; the fusion, not the tiling,
// is the point of this prototype).
template <class Epilogue, bool B_T, bool HAS_BIAS>
__global__ void gemm_epilogue_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     const float* __restrict__ bias,
                                     float* __restrict__ C,
                                     int Mrows, int Ncols, int Kinner,
                                     Epilogue ep) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= Mrows * Ncols) return;

    int i = idx / Ncols;
    int j = idx % Ncols;

    float acc = HAS_BIAS ? bias[j] : 0.0f;
    const float* arow = A + i * Kinner;
    for (int k = 0; k < Kinner; ++k) {
        float bkj = B_T ? B[j * Kinner + k] : B[k * Ncols + j];
        acc = fmaf(arow[k], bkj, acc);
    }
    C[idx] = ep(acc);           // fused epilogue, inlined into the store
}

template <class Epilogue, bool B_T, bool HAS_BIAS>
inline void launch_gemm_epilogue(const float* A, const float* B, const float* bias,
                                 float* C, int Mrows, int Ncols, int Kinner, Epilogue ep) {
    int total = Mrows * Ncols;
    if (total == 0) return;
    const int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    gemm_epilogue_kernel<Epilogue, B_T, HAS_BIAS>
        <<<grid, BLOCK>>>(A, B, bias, C, Mrows, Ncols, Kinner, ep);
}
