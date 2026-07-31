// Metal solution wrapper stub for "nvfp4-quantize" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const half* a, float sf_g, uint8_t* q, __nv_fp8_e4m3* scale, size_t m, size_t k) {
    auto pso = tensor::pipeline("nvfp4-quantize");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
