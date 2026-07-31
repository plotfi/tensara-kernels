#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k);

int main() {
    tensor::begin("mxfp8-dequantize");

    size_t m = 64;
    size_t k = 64;

    tensor::Buffer<uint8_t> q(m * k);
    tensor::Buffer<uint8_t> scale(m * (k / 32));
    tensor::Buffer<float> out(m * k);

    q.fill_random();
    scale.fill_random();

    BENCHMARK(solution(q, scale, out, m, k));

    out.preview("out");

    tensor::end();
}
