#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* predictions, const float* targets, float* output, const size_t* shape, size_t ndim);

int main(int argc, char** argv) {
    printf("=== mse-loss ===\n");
    srand(42);

    size_t ndim = 1;

    // Default shape: 2D tensor 64x64
    size_t h_shape[] = {64, 64};
    ndim = 2;

    size_t predictions_count = 64 * 64;
    size_t targets_count = 64 * 64;
    size_t output_count = 1;
    size_t shape_count = ndim;

    float* h_predictions = new float[predictions_count];
    float* h_targets = new float[targets_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < predictions_count; i++)
        h_predictions[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < targets_count; i++)
        h_targets[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_predictions;
    cudaMalloc(&d_predictions, predictions_count * sizeof(float));
    float* d_targets;
    cudaMalloc(&d_targets, targets_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));
    size_t* d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(size_t));
    cudaMemcpy(d_shape, h_shape, ndim * sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_predictions, h_predictions, predictions_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, targets_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    BENCHMARK(solution(d_predictions, d_targets, d_output, d_shape, ndim));

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_output);
    cudaFree(d_shape);
    delete[] h_predictions;
    delete[] h_targets;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}
