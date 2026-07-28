// Correctness test for matrix-power: out = M^n (integer matrix power).
// Harness uses n=3 (exponent), size=8. Reference multiplies M by itself n times.

#include "test_utils.cuh"
extern "C" void solution(const float* input_matrix, size_t n, float* output_matrix, size_t size);

static void matmul(const double* A, const double* B, double* C, size_t s) {
    for (size_t i = 0; i < s; i++)
        for (size_t j = 0; j < s; j++) {
            double acc = 0.0;
            for (size_t p = 0; p < s; p++) acc += A[i * s + p] * B[p * s + j];
            C[i * s + j] = acc;
        }
}

int main() {
    test::seed();
    size_t n = 3, size = 8;

    float* h_in  = new float[size * size];
    float* h_out = new float[size * size];
    float* h_ref = new float[size * size];
    test::fill_random(h_in, size * size);

    test::DBuf<float> d_in(size * size), d_out(size * size);
    d_in.upload(h_in);
    solution(d_in, n, d_out, size);
    test::check_cuda("matrix-power");
    d_out.download(h_out);

    // Reference: repeated multiply in double, then cast.
    double* acc  = new double[size * size];
    double* base = new double[size * size];
    double* tmp  = new double[size * size];
    for (size_t i = 0; i < size * size; i++) base[i] = h_in[i];
    // acc = identity
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++) acc[i * size + j] = (i == j) ? 1.0 : 0.0;
    for (size_t e = 0; e < n; e++) { matmul(acc, base, tmp, size); std::swap(acc, tmp); }
    for (size_t i = 0; i < size * size; i++) h_ref[i] = static_cast<float>(acc[i]);

    int bad = test::compare("matrix-power", h_out, h_ref, size * size, 1e-3f, 1e-4f);
    int rc = test::report("matrix-power", bad, size * size);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    delete[] acc; delete[] base; delete[] tmp;
    return rc;
}
