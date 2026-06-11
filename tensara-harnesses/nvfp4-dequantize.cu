#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint8_t* q, const __nv_fp8_e4m3* scale, float sf_g, float* out, size_t m, size_t k);

int main() {
    harness::begin("nvfp4-dequantize");

    float sf_g = 1.0f;
    size_t m = 64;
    size_t k = 64;

    harness::Buffer<uint8_t> q(m * k / 2);
    harness::Buffer<__nv_fp8_e4m3> scale(m * (k / 16));
    harness::Buffer<float> out(m * k);

    q.fill_random();
    scale.fill_random();

    BENCHMARK(solution(q, scale, sf_g, out, m, k));

    out.preview("out");

    harness::end();
}
