// Metal solution wrapper for "threshold" (unified activation kernel, 8 elems/thread).
// Mirrors the CUDA solution: alpha carries the threshold value.
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    auto pso = tensor::pipeline("threshold");
    uint32_t N = static_cast<uint32_t>(height * width);
    float alpha = threshold_value;
    size_t threads = (N + 7u) / 8u;
    tensor::dispatch(pso, { tensor::buf(input_image), tensor::buf(output_image) },
                      { tensor::arg(N), tensor::arg(alpha) }, threads);
}
