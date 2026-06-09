#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" void solution(const uint32_t* d_input1, const uint32_t* d_input2, uint32_t* d_output, size_t n);

int main(int argc, char** argv) {
    printf("=== vector-multiply-ff ===\n");
    srand(42);

    size_t n = 1024;

    size_t d_input1_count = n;
    size_t d_input2_count = n;
    size_t d_output_count = n;

    uint32_t* h_d_input1 = new uint32_t[d_input1_count];
    uint32_t* h_d_input2 = new uint32_t[d_input2_count];
    uint32_t* h_d_output = new uint32_t[d_output_count];

    for (size_t i = 0; i < d_input1_count; i++)
        h_d_input1[i] = static_cast<uint32_t>(rand());
    for (size_t i = 0; i < d_input2_count; i++)
        h_d_input2[i] = static_cast<uint32_t>(rand());
    memset(h_d_output, 0, d_output_count * sizeof(uint32_t));

    uint32_t* d_d_input1;
    cudaMalloc(&d_d_input1, d_input1_count * sizeof(uint32_t));
    uint32_t* d_d_input2;
    cudaMalloc(&d_d_input2, d_input2_count * sizeof(uint32_t));
    uint32_t* d_d_output;
    cudaMalloc(&d_d_output, d_output_count * sizeof(uint32_t));

    cudaMemcpy(d_d_input1, h_d_input1, d_input1_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_input2, h_d_input2, d_input2_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_d_output, 0, d_output_count * sizeof(uint32_t));

    solution(d_d_input1, d_d_input2, d_d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d_output, d_d_output, d_output_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Output d_output (first 10): ");
    for (size_t i = 0; i < 10 && i < d_output_count; i++)
        printf("%u ", h_d_output[i]);
    printf("\n");

    cudaFree(d_d_input1);
    cudaFree(d_d_input2);
    cudaFree(d_d_output);
    delete[] h_d_input1;
    delete[] h_d_input2;
    delete[] h_d_output;

    printf("Done.\n");
    return 0;
}
