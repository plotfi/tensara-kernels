#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint8_t* q, const __nv_fp8_e4m3* scale, float sf_g, float* out, size_t m, size_t k);

int main() {
    tensor::begin("nvfp4-dequantize");

    float sf_g = 1.0f;
    size_t m = tensor::bench_size("M", 64);
    size_t k = tensor::bench_size("K", 64);

    tensor::Buffer<uint8_t> q(m * k / 2);
    tensor::Buffer<__nv_fp8_e4m3> scale(m * (k / 16));
    tensor::Buffer<float> out(m * k);

    q.fill_random();
    scale.fill_random();

    BENCHMARK(solution(q, scale, sf_g, out, m, k));

    out.preview("out");

    tensor::end();
}
