// Metal solution wrapper for "relu" (unified activation kernel, 8 elems/thread).
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    auto pso = tensor::pipeline("relu");
    uint32_t N = static_cast<uint32_t>(n * m);
    float alpha = 0.0f;
    size_t threads = (N + 7u) / 8u;
    tensor::dispatch(pso, { tensor::buf(input), tensor::buf(output) },
                      { tensor::arg(N), tensor::arg(alpha) }, threads);
}
