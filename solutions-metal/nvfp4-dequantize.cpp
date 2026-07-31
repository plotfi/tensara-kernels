// Metal solution wrapper stub for "nvfp4-dequantize" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint8_t* q, const __nv_fp8_e4m3* scale, float sf_g, float* out, size_t m, size_t k) {
    auto pso = tensor::pipeline("nvfp4-dequantize");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
