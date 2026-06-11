#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* a, uint8_t* q, uint8_t* scale, size_t m, size_t k);

int main() {
    harness::begin("mxfp8-quantize");

    size_t m = 64;
    size_t k = 64;

    harness::Buffer<float> a(m * k);
    harness::Buffer<uint8_t> q(m * k);
    harness::Buffer<uint8_t> scale(m * (k / 32));

    a.fill_random();

    BENCHMARK(solution(a, q, scale, m, k));

    q.preview("q");
    scale.preview("scale");

    harness::end();
}
