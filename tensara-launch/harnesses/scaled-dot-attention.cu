#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const float* Q, const float* K, const float* V, float* output, size_t B, size_t H, size_t S, size_t E);

int main(int argc, char** argv) {
    printf("=== scaled-dot-attention ===\n");
    srand(42);

    size_t B = 2;
    size_t H = 4;
    size_t S = 32;
    size_t E = 64;

    size_t Q_count = B * H * S * E;
    size_t K_count = B * H * S * E;
    size_t V_count = B * H * S * E;
    size_t output_count = B * H * S * E;

    float* h_Q = new float[Q_count];
    float* h_K = new float[K_count];
    float* h_V = new float[V_count];
    float* h_output = new float[output_count];

    for (size_t i = 0; i < Q_count; i++)
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < K_count; i++)
        h_K[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < V_count; i++)
        h_V[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    memset(h_output, 0, output_count * sizeof(float));

    float* d_Q;
    cudaMalloc(&d_Q, Q_count * sizeof(float));
    float* d_K;
    cudaMalloc(&d_K, K_count * sizeof(float));
    float* d_V;
    cudaMalloc(&d_V, V_count * sizeof(float));
    float* d_output;
    cudaMalloc(&d_output, output_count * sizeof(float));

    cudaMemcpy(d_Q, h_Q, Q_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, K_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, V_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_count * sizeof(float));

    solution(d_Q, d_K, d_V, d_output, B, H, S, E);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, output_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output output (first 10): ");
    for (size_t i = 0; i < 10 && i < output_count; i++)
        printf("%f ", h_output[i]);
    printf("\n");

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_output;

    printf("Done.\n");
    return 0;
}
