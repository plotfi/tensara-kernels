// Metal solution wrapper stub for "mxfp8-dequantize" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k) {
    auto pso = tensor::pipeline("mxfp8-dequantize");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
