// Correctness test for cumsum (inclusive prefix sum): out[i] = sum_{j<=i} in[j].

#include "test_utils.cuh"
extern "C" void solution(const float* input, float* output, size_t N);

int main() {
    test::seed();
    size_t N = 1024;

    float* h_in  = new float[N];
    float* h_out = new float[N];
    float* h_ref = new float[N];
    test::fill_random(h_in, N);

    test::DBuf<float> d_in(N), d_out(N);
    d_in.upload(h_in);
    solution(d_in, d_out, N);
    test::check_cuda("cumsum");
    d_out.download(h_out);

    double acc = 0.0;
    for (size_t i = 0; i < N; i++) { acc += h_in[i]; h_ref[i] = static_cast<float>(acc); }

    int bad = test::compare("cumsum", h_out, h_ref, N, 1e-3f, 1e-4f);
    int rc = test::report("cumsum", bad, N);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
