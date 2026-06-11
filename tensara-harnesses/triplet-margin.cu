#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin);

int main() {
    harness::begin("triplet-margin");

    size_t B = 8;
    size_t E = 128;
    float margin = 1.0f;

    harness::Buffer<float> anchor(B * E);
    harness::Buffer<float> positive(B * E);
    harness::Buffer<float> negative(B * E);
    harness::Buffer<float> loss(1);

    anchor.fill_random();
    positive.fill_random();
    negative.fill_random();

    BENCHMARK(solution(anchor, positive, negative, loss, B, E, margin));

    loss.preview("loss");

    printf("Done.\n");
    return 0;
}
