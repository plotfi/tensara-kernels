// Correctness test for average pooling, dimension selected at compile time:
//   -DPOOL_DIM=1 -> avg-pool-1d,  -DPOOL_DIM=2 -> avg-pool-2d,  -DPOOL_DIM=3 -> avg-pool-3d
//
// ASSUMED semantics (PyTorch AvgPool default, count_include_pad=True): a window
// starting at o*stride - padding, summing zeros for out-of-range taps, divided
// by the full window size (kernel_size^DIM). Output length per axis is
// floor((L + 2*padding - kernel_size)/stride) + 1.

#include "test_utils.cuh"

#ifndef POOL_DIM
#define POOL_DIM 1
#endif

#if POOL_DIM == 1
#  define POOL_NAME "avg-pool-1d"
extern "C" void solution(const float* input, int kernel_size, int stride, int padding,
                         float* output, size_t H);
#elif POOL_DIM == 2
#  define POOL_NAME "avg-pool-2d"
extern "C" void solution(const float* input, int kernel_size, int stride, int padding,
                         float* output, size_t H, size_t W);
#else
#  define POOL_NAME "avg-pool-3d"
extern "C" void solution(const float* input, int kernel_size, int stride, int padding,
                         float* output, size_t H, size_t W, size_t D);
#endif

static int out_len(int L, int ks, int stride, int pad) {
    return (L + 2 * pad - ks) / stride + 1;
}

int main() {
    test::seed();
    int ks = 3, stride = 1, pad = 1;

#if POOL_DIM == 1
    int H = 1024;
    int OH = out_len(H, ks, stride, pad);
    size_t in_n = H, out_n = OH;
#elif POOL_DIM == 2
    int H = 32, W = 32;
    int OH = out_len(H, ks, stride, pad), OW = out_len(W, ks, stride, pad);
    size_t in_n = (size_t)H * W, out_n = (size_t)OH * OW;
#else
    int H = 16, W = 16, D = 16;
    int OH = out_len(H, ks, stride, pad), OW = out_len(W, ks, stride, pad), OD = out_len(D, ks, stride, pad);
    size_t in_n = (size_t)H * W * D, out_n = (size_t)OH * OW * OD;
#endif

    float* h_in  = new float[in_n];
    float* h_out = new float[out_n];
    float* h_ref = new float[out_n];
    test::fill_random(h_in, in_n);

    test::DBuf<float> d_in(in_n), d_out(out_n);
    d_in.upload(h_in);
#if POOL_DIM == 1
    solution(d_in, ks, stride, pad, d_out, H);
#elif POOL_DIM == 2
    solution(d_in, ks, stride, pad, d_out, H, W);
#else
    solution(d_in, ks, stride, pad, d_out, H, W, D);
#endif
    test::check_cuda(POOL_NAME);
    d_out.download(h_out);

    double denom = 1.0;
    for (int d = 0; d < POOL_DIM; d++) denom *= ks;

#if POOL_DIM == 1
    for (int o = 0; o < OH; o++) {
        double acc = 0.0;
        for (int k = 0; k < ks; k++) {
            int ii = o * stride - pad + k;
            if (ii >= 0 && ii < H) acc += h_in[ii];
        }
        h_ref[o] = static_cast<float>(acc / denom);
    }
#elif POOL_DIM == 2
    for (int oi = 0; oi < OH; oi++)
        for (int oj = 0; oj < OW; oj++) {
            double acc = 0.0;
            for (int ki = 0; ki < ks; ki++)
                for (int kj = 0; kj < ks; kj++) {
                    int ii = oi * stride - pad + ki, jj = oj * stride - pad + kj;
                    if (ii >= 0 && ii < H && jj >= 0 && jj < W) acc += h_in[ii * W + jj];
                }
            h_ref[oi * OW + oj] = static_cast<float>(acc / denom);
        }
#else
    for (int oi = 0; oi < OH; oi++)
        for (int oj = 0; oj < OW; oj++)
            for (int ok = 0; ok < OD; ok++) {
                double acc = 0.0;
                for (int ki = 0; ki < ks; ki++)
                    for (int kj = 0; kj < ks; kj++)
                        for (int kk = 0; kk < ks; kk++) {
                            int ii = oi * stride - pad + ki, jj = oj * stride - pad + kj, ll = ok * stride - pad + kk;
                            if (ii >= 0 && ii < H && jj >= 0 && jj < W && ll >= 0 && ll < D)
                                acc += h_in[(ii * W + jj) * D + ll];
                        }
                h_ref[(oi * OW + oj) * OD + ok] = static_cast<float>(acc / denom);
            }
#endif

    int bad = test::compare(POOL_NAME, h_out, h_ref, out_n, 1e-4f, 1e-5f);
    int rc = test::report(POOL_NAME, bad, out_n);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
