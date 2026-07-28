// Correctness test for cosine-similarity along each row (n rows of length d):
//   output[i] = dot(p_i, t_i) / (||p_i|| * ||t_i||)

#include "test_utils.cuh"
extern "C" void solution(const float* predictions, const float* targets, float* output,
                         size_t n, size_t d);

int main() {
    test::seed();
    size_t n = 64, d = 128;

    float* h_p = new float[n * d];
    float* h_t = new float[n * d];
    float* h_o = new float[n];
    float* h_ref = new float[n];
    test::fill_random(h_p, n * d);
    test::fill_random(h_t, n * d);

    test::DBuf<float> d_p(n * d), d_t(n * d), d_o(n);
    d_p.upload(h_p); d_t.upload(h_t);
    solution(d_p, d_t, d_o, n, d);
    test::check_cuda("cosine-similarity");
    d_o.download(h_o);

    for (size_t i = 0; i < n; i++) {
        double dot = 0.0, np = 0.0, nt = 0.0;
        for (size_t j = 0; j < d; j++) {
            double a = h_p[i * d + j], b = h_t[i * d + j];
            dot += a * b; np += a * a; nt += b * b;
        }
        h_ref[i] = static_cast<float>(dot / (sqrt(np) * sqrt(nt)));
    }

    int bad = test::compare("cosine-similarity", h_o, h_ref, n, 1e-3f, 1e-4f);
    int rc = test::report("cosine-similarity", bad, n);
    delete[] h_p; delete[] h_t; delete[] h_o; delete[] h_ref;
    return rc;
}
