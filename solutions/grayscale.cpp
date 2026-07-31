// Metal solution wrapper for "grayscale".
#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels) {
    auto pso = tensor::pipeline("grayscale");
    uint32_t N = static_cast<uint32_t>(height * width);
    tensor::dispatch(pso, { tensor::buf(rgb_image), tensor::buf(grayscale_output) },
                      { tensor::arg(N) }, height * width);
}
