#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* Q, const float* K, const float* V, float* output, size_t B, size_t H, size_t S, size_t E);

int main() {
    harness::begin("scaled-dot-attention");

    size_t B = 2;
    size_t H = 4;
    size_t S = 32;
    size_t E = 64;

    harness::Buffer<float> Q(B * H * S * E);
    harness::Buffer<float> K(B * H * S * E);
    harness::Buffer<float> V(B * H * S * E);
    harness::Buffer<float> output(B * H * S * E);

    Q.fill_random();
    K.fill_random();
    V.fill_random();

    BENCHMARK(solution(Q, K, V, output, B, H, S, E));

    output.preview("output");

    harness::end();
}
