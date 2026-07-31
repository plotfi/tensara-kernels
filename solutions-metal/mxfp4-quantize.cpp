// Metal solution wrapper stub for "mxfp4-quantize" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* a, uint8_t* q, uint8_t* scale, size_t m, size_t k) {
    auto pso = tensor::pipeline("mxfp4-quantize");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
