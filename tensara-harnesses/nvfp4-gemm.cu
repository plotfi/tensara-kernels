#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint8_t* q_a, const __nv_fp8_e4m3* scale_a, float sf_g_a, const uint8_t* q_b, const __nv_fp8_e4m3* scale_b, float sf_g_b, half* c, size_t m, size_t n, size_t k);

int main() {
    harness::begin("nvfp4-gemm");

    float sf_g_a = 1.0f;
    float sf_g_b = 1.0f;
    size_t m = 64;
    size_t n = 64;
    size_t k = 64;

    harness::Buffer<uint8_t> q_a(m * k / 2);
    harness::Buffer<__nv_fp8_e4m3> scale_a(m * (k / 16));
    harness::Buffer<uint8_t> q_b(k * n / 2);
    harness::Buffer<__nv_fp8_e4m3> scale_b(k * (n / 16));
    harness::Buffer<half> c(m * n);

    q_a.fill_random();
    scale_a.fill_random();
    q_b.fill_random();
    scale_b.fill_random();

    BENCHMARK(solution(q_a, scale_a, sf_g_a, q_b, scale_b, sf_g_b, c, m, n, k));

    c.preview("c");

    printf("Done.\n");
    return 0;
}
