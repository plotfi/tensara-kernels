#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k);

int main() {
    harness::begin("mxfp4-dequantize");

    size_t m = 64;
    size_t k = 64;

    harness::Buffer<uint8_t> q(m * k / 2);
    harness::Buffer<uint8_t> scale(m * (k / 32));
    harness::Buffer<float> out(m * k);

    q.fill_random();
    scale.fill_random();

    BENCHMARK(solution(q, scale, out, m, k));

    out.preview("out");

    harness::end();
}
