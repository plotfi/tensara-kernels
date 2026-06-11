#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* X, float* Y, size_t B, size_t D);

int main(int argc, char** argv) {
    printf("=== l2-norm ===\n");
    srand(42);

    size_t B = 8;
    size_t D = 64;

    size_t X_count = B * D;
    size_t Y_count = B * D;

    float* h_X = new float[X_count];
    float* h_Y = new float[Y_count];

    for (size_t i = 0; i < X_count; i++)
        h_X[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_Y, 0, Y_count * sizeof(float));

    float* d_X;
    cudaMalloc(&d_X, X_count * sizeof(float));
    float* d_Y;
    cudaMalloc(&d_Y, Y_count * sizeof(float));

    cudaMemcpy(d_X, h_X, X_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_Y, 0, Y_count * sizeof(float));

    BENCHMARK(solution(d_X, d_Y, B, D));

    cudaMemcpy(h_Y, d_Y, Y_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output Y (first 10): ");
    for (size_t i = 0; i < 10 && i < Y_count; i++)
        printf("%f ", h_Y[i]);
    printf("\n");

    cudaFree(d_X);
    cudaFree(d_Y);
    delete[] h_X;
    delete[] h_Y;

    printf("Done.\n");
    return 0;
}
