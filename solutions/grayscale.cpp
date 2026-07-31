// Metal solution wrapper for "grayscale".
#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels) {
    auto pso = harness::pipeline("grayscale");
    uint32_t N = static_cast<uint32_t>(height * width);
    harness::dispatch(pso, { harness::buf(rgb_image), harness::buf(grayscale_output) },
                      { harness::arg(N) }, height * width);
}
