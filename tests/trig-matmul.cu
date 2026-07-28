// Correctness test for upper-/lower-triangular matmul: C = A * B where A, B are
// triangular. Selected at compile time:
//   -DTRIG_UPPER  -> upper-trig-matmul
//   -DTRIG_LOWER  -> lower-trig-matmul
// The test masks the inputs to the chosen triangle so the result is unambiguous,
// then compares against a full matmul (the opposite triangle should be ~0).

#include "test_utils.cuh"
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n);

#if defined(TRIG_LOWER)
#  define TRIG_NAME "lower-trig-matmul"
#  define KEEP(i, j) ((i) >= (j))   // keep lower triangle
#else
#  define TRIG_NAME "upper-trig-matmul"
#  define KEEP(i, j) ((i) <= (j))   // keep upper triangle
#endif

int main() {
    test::seed();
    size_t n = 64;

    float* h_a = new float[n * n];
    float* h_b = new float[n * n];
    float* h_c = new float[n * n];
    float* h_ref = new float[n * n];
    test::fill_random(h_a, n * n);
    test::fill_random(h_b, n * n);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            if (!KEEP(i, j)) { h_a[i * n + j] = 0.0f; h_b[i * n + j] = 0.0f; }

    test::DBuf<float> d_a(n * n), d_b(n * n), d_c(n * n);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_c, n);
    test::check_cuda(TRIG_NAME);
    d_c.download(h_c);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (size_t p = 0; p < n; p++)
                acc += static_cast<double>(h_a[i * n + p]) * h_b[p * n + j];
            // A kernel may only write its triangle; the other side is 0 in ref too.
            h_ref[i * n + j] = KEEP(i, j) ? static_cast<float>(acc) : 0.0f;
        }

    int bad = test::compare(TRIG_NAME, h_c, h_ref, n * n, 1e-3f, 1e-3f);
    int rc = test::report(TRIG_NAME, bad, n * n);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
