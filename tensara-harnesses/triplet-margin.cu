#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin);

int main(int argc, char** argv) {
    printf("=== triplet-margin ===\n");
    srand(42);

    size_t B = 8;
    size_t E = 128;
    float margin = 1.0f;

    size_t anchor_count = B * E;
    size_t positive_count = B * E;
    size_t negative_count = B * E;
    size_t loss_count = 1;

    float* h_anchor = new float[anchor_count];
    float* h_positive = new float[positive_count];
    float* h_negative = new float[negative_count];
    float* h_loss = new float[loss_count];

    for (size_t i = 0; i < anchor_count; i++)
        h_anchor[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < positive_count; i++)
        h_positive[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < negative_count; i++)
        h_negative[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_loss, 0, loss_count * sizeof(float));

    float* d_anchor;
    cudaMalloc(&d_anchor, anchor_count * sizeof(float));
    float* d_positive;
    cudaMalloc(&d_positive, positive_count * sizeof(float));
    float* d_negative;
    cudaMalloc(&d_negative, negative_count * sizeof(float));
    float* d_loss;
    cudaMalloc(&d_loss, loss_count * sizeof(float));

    cudaMemcpy(d_anchor, h_anchor, anchor_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, h_positive, positive_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, h_negative, negative_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_loss, 0, loss_count * sizeof(float));

    BENCHMARK(solution(d_anchor, d_positive, d_negative, d_loss, B, E, margin));

    cudaMemcpy(h_loss, d_loss, loss_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output loss (first 10): ");
    for (size_t i = 0; i < 10 && i < loss_count; i++)
        printf("%f ", h_loss[i]);
    printf("\n");

    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_loss);
    delete[] h_anchor;
    delete[] h_positive;
    delete[] h_negative;
    delete[] h_loss;

    printf("Done.\n");
    return 0;
}
