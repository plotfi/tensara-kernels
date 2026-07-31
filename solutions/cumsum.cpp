// Metal solution wrapper stub for "cumsum" (unimplemented).
// Dispatches the no-op stub shader. Once implemented, bind the real
// buffers (tensor::buf(ptr)) and scalars (tensor::arg(v)) and dispatch
// over the output, mirroring the CUDA solution.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, float* output, size_t N) {
    auto pso = tensor::pipeline("cumsum");
    tensor::dispatch(pso, {}, 1);  // TODO: bind real buffers/scalars
}
