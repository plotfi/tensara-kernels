#include "../tensor-lib/tensor.cuh"

extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin);

int main() {
    tensor::begin("triplet-margin");

    size_t B = 8;
    size_t E = 128;
    float margin = 1.0f;

    tensor::Buffer<float> anchor(B * E);
    tensor::Buffer<float> positive(B * E);
    tensor::Buffer<float> negative(B * E);
    tensor::Buffer<float> loss(1);

    anchor.fill_random();
    positive.fill_random();
    negative.fill_random();

    BENCHMARK(solution(anchor, positive, negative, loss, B, E, margin));

    loss.preview("loss");

    tensor::end();
}
