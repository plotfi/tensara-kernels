// Metal solution wrapper stub for "ecc-point-negation" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint64_t* xs, const uint64_t* ys, uint64_t p, uint64_t* out_xy, size_t n) {
    auto pso = tensor::pipeline("ecc-point-negation");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
