#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint8_t* q_a, const __nv_fp8_e4m3* scale_a, float sf_g_a, const uint8_t* q_x, const __nv_fp8_e4m3* scale_x, float sf_g_x, half* y, size_t m, size_t k);

int main() {
    tensor::begin("nvfp4-gemv");

    float sf_g_a = 1.0f;
    float sf_g_x = 1.0f;
    size_t m = 64;
    size_t k = 64;

    tensor::Buffer<uint8_t> q_a(m * k / 2);
    tensor::Buffer<__nv_fp8_e4m3> scale_a(m * (k / 16));
    tensor::Buffer<uint8_t> q_x(k / 2);
    tensor::Buffer<__nv_fp8_e4m3> scale_x(k / 16);
    tensor::Buffer<half> y(m);

    q_a.fill_random();
    scale_a.fill_random();
    q_x.fill_random();
    scale_x.fill_random();

    BENCHMARK(solution(q_a, scale_a, sf_g_a, q_x, scale_x, sf_g_x, y, m, k));

    y.preview("y");

    tensor::end();
}
