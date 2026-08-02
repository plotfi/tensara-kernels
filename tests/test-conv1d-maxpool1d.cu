// Correctness test for the fused conv1d -> maxpool1d kernel.
// Reference: same-length centered cross-correlation (conv[c] = sum_j in[c+j-rc]*w[j],
// rc=(K-1)/2, zero-padded), then max pool over conv[] with (ks,stride,pad,dil);
// padded (out-of-range) pool taps are ignored. Output length is
// floor((N + 2*pad - dil*(ks-1) - 1)/stride) + 1.

#include "test_utils.cuh"
extern "C" void solution(const float* input, const float* weight, float* output,
                         size_t N, size_t K,
                         int kernel_size, int stride, int padding, int dilation);

int main() {
    test::seed();
    int N = 1024, K = 5;
    int ks = 3, stride = 2, pad = 1, dil = 1;
    int rc = (K - 1) / 2;
    int Lout = (N + 2 * pad - dil * (ks - 1) - 1) / stride + 1;

    float* h_in  = new float[N];
    float* h_w   = new float[K];
    float* h_out = new float[Lout];
    float* h_ref = new float[Lout];
    float* conv  = new float[N];
    test::fill_random(h_in, N);
    test::fill_random(h_w, K);

    test::DBuf<float> d_in(N), d_w(K), d_out(Lout);
    d_in.upload(h_in); d_w.upload(h_w);
    solution(d_in, d_w, d_out, N, K, ks, stride, pad, dil);
    test::check_cuda("conv1d-maxpool1d");
    d_out.download(h_out);

    // producer: same-length centered conv
    for (int c = 0; c < N; c++) {
        double acc = 0.0;
        for (int j = 0; j < K; j++) {
            int xi = c + j - rc;
            if (xi >= 0 && xi < N) acc += static_cast<double>(h_in[xi]) * h_w[j];
        }
        conv[c] = static_cast<float>(acc);
    }
    // consumer: max pool over conv
    for (int o = 0; o < Lout; o++) {
        float m = -INFINITY;
        for (int k = 0; k < ks; k++) {
            int c = o * stride + k * dil - pad;
            if (c >= 0 && c < N) m = fmaxf(m, conv[c]);
        }
        h_ref[o] = m;
    }

    int bad = test::compare("conv1d-maxpool1d", h_out, h_ref, Lout, 1e-3f, 1e-3f);
    int rcode = test::report("conv1d-maxpool1d", bad, Lout);
    delete[] h_in; delete[] h_w; delete[] h_out; delete[] h_ref; delete[] conv;
    return rcode;
}
