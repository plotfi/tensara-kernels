// Correctness test for cs4803dgc-lab2-matmul (C = A * B, square 256x256) vs a CPU gold
// (matches CS4803VGCD/lab2/matrixmul_gold.cpp).
#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, float* C, size_t n);

int main() {
    test::seed();
    size_t n = 256;
    float* hA = new float[n*n]; float* hB = new float[n*n];
    float* hC = new float[n*n]; float* hRef = new float[n*n];
    test::fill_random(hA, n*n); test::fill_random(hB, n*n);

    test::DBuf<float> dA(n*n), dB(n*n), dC(n*n);
    dA.upload(hA); dB.upload(hB);
    solution(dA, dB, dC, n);
    test::check_cuda("cs4803dgc-lab2-matmul");
    dC.download(hC);

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0;
            for (size_t k = 0; k < n; ++k) s += (double)hA[i*n+k] * hB[k*n+j];
            hRef[i*n+j] = (float)s;
        }

    int bad = test::compare("cs4803dgc-lab2-matmul", hC, hRef, n*n, 1e-3f, 1e-4f);
    int rc = test::report("cs4803dgc-lab2-matmul", bad, n*n);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    return rc;
}
