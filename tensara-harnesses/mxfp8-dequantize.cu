#include "../kernel-implementation/harness.cuh"

extern "C" void solution(const uint8_t* q, const uint8_t* scale, float* out, size_t m, size_t k);

int main(int argc, char** argv) {
    printf("=== mxfp8-dequantize ===\n");
    srand(42);

    size_t m = 64;
    size_t k = 64;

    size_t q_count = m * k;
    size_t scale_count = m * (k / 32);
    size_t out_count = m * k;

    uint8_t* h_q = new uint8_t[q_count];
    uint8_t* h_scale = new uint8_t[scale_count];
    float* h_out = new float[out_count];

    for (size_t i = 0; i < q_count; i++)
        h_q[i] = static_cast<uint8_t>(rand() % 256);
    for (size_t i = 0; i < scale_count; i++)
        h_scale[i] = static_cast<uint8_t>(rand() % 256);
    memset(h_out, 0, out_count * sizeof(float));

    uint8_t* d_q;
    cudaMalloc(&d_q, q_count * sizeof(uint8_t));
    uint8_t* d_scale;
    cudaMalloc(&d_scale, scale_count * sizeof(uint8_t));
    float* d_out;
    cudaMalloc(&d_out, out_count * sizeof(float));

    cudaMemcpy(d_q, h_q, q_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, h_scale, scale_count * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_count * sizeof(float));

    BENCHMARK(solution(d_q, d_scale, d_out, m, k));

    cudaMemcpy(h_out, d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output out (first 10): ");
    for (size_t i = 0; i < 10 && i < out_count; i++)
        printf("%f ", h_out[i]);
    printf("\n");

    cudaFree(d_q);
    cudaFree(d_scale);
    cudaFree(d_out);
    delete[] h_q;
    delete[] h_scale;
    delete[] h_out;

    printf("Done.\n");
    return 0;
}
