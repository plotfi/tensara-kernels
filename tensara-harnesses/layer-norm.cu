#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const float* X, const float* gamma, const float* beta, float* Y, size_t B, size_t F, size_t D1, size_t D2);

int main(int argc, char** argv) {
    printf("=== layer-norm ===\n");
    srand(42);

    size_t B = 2;
    size_t F = 4;
    size_t D1 = 8;
    size_t D2 = 8;

    size_t X_count = B * F * D1 * D2;
    size_t gamma_count = F * D1 * D2;
    size_t beta_count = F * D1 * D2;
    size_t Y_count = B * F * D1 * D2;

    float* h_X = new float[X_count];
    float* h_gamma = new float[gamma_count];
    float* h_beta = new float[beta_count];
    float* h_Y = new float[Y_count];

    for (size_t i = 0; i < X_count; i++)
        h_X[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < gamma_count; i++)
        h_gamma[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < beta_count; i++)
        h_beta[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_Y, 0, Y_count * sizeof(float));

    float* d_X;
    cudaMalloc(&d_X, X_count * sizeof(float));
    float* d_gamma;
    cudaMalloc(&d_gamma, gamma_count * sizeof(float));
    float* d_beta;
    cudaMalloc(&d_beta, beta_count * sizeof(float));
    float* d_Y;
    cudaMalloc(&d_Y, Y_count * sizeof(float));

    cudaMemcpy(d_X, h_X, X_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, gamma_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, beta_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_Y, 0, Y_count * sizeof(float));

    BENCHMARK(solution(d_X, d_gamma, d_beta, d_Y, B, F, D1, D2));

    cudaMemcpy(h_Y, d_Y, Y_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output Y (first 10): ");
    for (size_t i = 0; i < 10 && i < Y_count; i++)
        printf("%f ", h_Y[i]);
    printf("\n");

    cudaFree(d_X);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_Y);
    delete[] h_X;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_Y;

    printf("Done.\n");
    return 0;
}
