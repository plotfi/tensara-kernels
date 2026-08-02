#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Prologue-fused GEMM -- the input-side mirror of gemm-epilogue.cuh.
//
// A *prologue* fuses a pointwise op onto the matmul's INPUTS at the load, before
// they enter the math (vs an epilogue, which acts on the OUTPUT at the store).
// The canonical use is a weight-only-quantized linear layer: the weight is
// stored quantized and must be DEQUANTIZED before the matmul. Fusing the dequant
// into the load means the full-precision weight is never materialized in global
// memory -- you read the small quantized tensor and expand it in-register at the
// point of use. See FUSION_NOTES.md ("Prologue fusion").
//
//   C[m,n] = sum_k  A[m,k] * dequant(Wq[n,k], scale[n, k/G])
//
// This is group-wise int8 quant (the GPTQ/AWQ shape): the weight row n is split
// into groups of G along K, each group with its own fp32 scale. Because the
// scale VARIES ALONG THE REDUCTION k, it cannot be hoisted out of the sum into
// an epilogue -- the dequant genuinely has to happen at the load. That's what
// makes this a prologue and not just an output scaling.
//
// Bandwidth win: Wq is int8 (1 byte) vs a materialized fp32 weight (4 bytes) --
// 4x less weight traffic, and no separate dequant kernel / intermediate buffer.
//
// Cost (the prologue tradeoff): this simple one-thread-per-output kernel re-reads
// and re-dequantizes each Wq[n,k] for every row m of A. A tiled kernel would
// dequant each weight tile once into shared memory and reuse it across the block
// -- the same recompute-vs-stage tradeoff as the Case-2 stencil fusion. Kept
// simple here so the fused-dequant load is the only moving part.

// The prologue op: dequantize one int8 weight with its (group) scale. Mirrors the
// epilogue functors in scalar-ops.cuh, but on the input side -- it takes the raw
// loaded operand plus its metadata and returns the value the math consumes.
struct DequantInt8 {
    __device__ __forceinline__ float operator()(int8_t q, float scale) const {
        return static_cast<float>(q) * scale;    // symmetric int8, zero-point 0
    }
};

template <class Prologue>
__global__ void gemm_prologue_kernel(const float* __restrict__ A,     // [M, K] activations
                                     const int8_t* __restrict__ Wq,   // [N, K] quantized weights
                                     const float* __restrict__ scale, // [N, K/G] group scales
                                     float* __restrict__ C,           // [M, N] output
                                     int M, int N, int K, int G,
                                     Prologue pro) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= M * N) return;
    int m = idx / N, n = idx % N;

    const float*  arow = A     + static_cast<size_t>(m) * K;
    const int8_t* wrow = Wq    + static_cast<size_t>(n) * K;
    const float*  srow = scale + static_cast<size_t>(n) * (K / G);

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float w = pro(wrow[k], srow[k / G]);   // PROLOGUE: dequant fused into the load
        acc = fmaf(arow[k], w, acc);           // fp32 weight `w` lives only in a register
    }
    C[idx] = acc;
}

template <class Prologue = DequantInt8>
inline void launch_gemm_prologue(const float* A, const int8_t* Wq, const float* scale,
                                 float* C, int M, int N, int K, int G, Prologue pro = {}) {
    int total = M * N;
    if (total == 0) return;
    const int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    gemm_prologue_kernel<Prologue><<<grid, BLOCK>>>(A, Wq, scale, C, M, N, K, G, pro);
}
