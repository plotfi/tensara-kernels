// Correctness test for running-sum-1d with window W=5.
// ASSUMED semantics: centered sliding-window sum with radius r=(W-1)/2, matching
// the repo's centered conv-1d convention; out-of-range taps contribute 0.
//   out[i] = sum_{j=-r..r, 0<=i+j<N} in[i+j]
// If your problem uses a trailing window (sum of the last W elements), adjust.

#include "test_utils.cuh"
extern "C" void solution(const float* input, size_t W, float* output, size_t N);

int main() {
    test::seed();
    size_t W = 5, N = 1024;

    float* h_in  = new float[N];
    float* h_out = new float[N];
    float* h_ref = new float[N];
    test::fill_random(h_in, N);

    test::DBuf<float> d_in(N), d_out(N);
    d_in.upload(h_in);
    solution(d_in, W, d_out, N);
    test::check_cuda("running-sum-1d");
    d_out.download(h_out);

    long r = (W - 1) / 2;
    for (long i = 0; i < (long)N; i++) {
        double acc = 0.0;
        for (long j = -r; j <= r; j++) {
            long k = i + j;
            if (k >= 0 && k < (long)N) acc += h_in[k];
        }
        h_ref[i] = static_cast<float>(acc);
    }

    int bad = test::compare("running-sum-1d", h_out, h_ref, N, 1e-4f, 1e-5f);
    int rc = test::report("running-sum-1d", bad, N);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
