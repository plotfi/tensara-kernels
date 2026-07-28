// Correctness test for max pooling, dimension selected at compile time:
//   -DPOOL_DIM=1 -> max-pool-1d,  -DPOOL_DIM=2 -> max-pool-2d,  -DPOOL_DIM=3 -> max-pool-3d
//
// ASSUMED semantics (PyTorch MaxPool): a window starting at o*stride - padding
// with dilation between taps; padded (out-of-range) positions are ignored
// (treated as -inf). Output length per axis is
// floor((L + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1.

#include "test_utils.cuh"

#ifndef POOL_DIM
#define POOL_DIM 1
#endif

#if POOL_DIM == 1
#  define POOL_NAME "max-pool-1d"
extern "C" void solution(const float* input, int kernel_size, int stride, int padding,
                         int dilation, float* output, size_t H);
#elif POOL_DIM == 2
#  define POOL_NAME "max-pool-2d"
extern "C" void solution(const float* input, int kernel_size, int stride, int padding,
                         int dilation, float* output, size_t H, size_t W);
#else
#  define POOL_NAME "max-pool-3d"
extern "C" void solution(const float* input, int kernel_size, int stride, int padding,
                         int dilation, float* output, size_t H, size_t W, size_t D);
#endif

static int out_len(int L, int ks, int stride, int pad, int dil) {
    return (L + 2 * pad - dil * (ks - 1) - 1) / stride + 1;
}

int main() {
    test::seed();
    int ks = 3, stride = 1, pad = 1, dil = 1;

#if POOL_DIM == 1
    int H = 1024;
    int OH = out_len(H, ks, stride, pad, dil);
    size_t in_n = H, out_n = OH;
#elif POOL_DIM == 2
    int H = 32, W = 32;
    int OH = out_len(H, ks, stride, pad, dil), OW = out_len(W, ks, stride, pad, dil);
    size_t in_n = (size_t)H * W, out_n = (size_t)OH * OW;
#else
    int H = 16, W = 16, D = 16;
    int OH = out_len(H, ks, stride, pad, dil), OW = out_len(W, ks, stride, pad, dil), OD = out_len(D, ks, stride, pad, dil);
    size_t in_n = (size_t)H * W * D, out_n = (size_t)OH * OW * OD;
#endif

    float* h_in  = new float[in_n];
    float* h_out = new float[out_n];
    float* h_ref = new float[out_n];
    test::fill_random(h_in, in_n);

    test::DBuf<float> d_in(in_n), d_out(out_n);
    d_in.upload(h_in);
#if POOL_DIM == 1
    solution(d_in, ks, stride, pad, dil, d_out, H);
#elif POOL_DIM == 2
    solution(d_in, ks, stride, pad, dil, d_out, H, W);
#else
    solution(d_in, ks, stride, pad, dil, d_out, H, W, D);
#endif
    test::check_cuda(POOL_NAME);
    d_out.download(h_out);

#if POOL_DIM == 1
    for (int o = 0; o < OH; o++) {
        float mx = -INFINITY;
        for (int k = 0; k < ks; k++) {
            int ii = o * stride - pad + k * dil;
            if (ii >= 0 && ii < H) mx = fmaxf(mx, h_in[ii]);
        }
        h_ref[o] = mx;
    }
#elif POOL_DIM == 2
    for (int oi = 0; oi < OH; oi++)
        for (int oj = 0; oj < OW; oj++) {
            float mx = -INFINITY;
            for (int ki = 0; ki < ks; ki++)
                for (int kj = 0; kj < ks; kj++) {
                    int ii = oi * stride - pad + ki * dil, jj = oj * stride - pad + kj * dil;
                    if (ii >= 0 && ii < H && jj >= 0 && jj < W) mx = fmaxf(mx, h_in[ii * W + jj]);
                }
            h_ref[oi * OW + oj] = mx;
        }
#else
    for (int oi = 0; oi < OH; oi++)
        for (int oj = 0; oj < OW; oj++)
            for (int ok = 0; ok < OD; ok++) {
                float mx = -INFINITY;
                for (int ki = 0; ki < ks; ki++)
                    for (int kj = 0; kj < ks; kj++)
                        for (int kk = 0; kk < ks; kk++) {
                            int ii = oi * stride - pad + ki * dil, jj = oj * stride - pad + kj * dil, ll = ok * stride - pad + kk * dil;
                            if (ii >= 0 && ii < H && jj >= 0 && jj < W && ll >= 0 && ll < D)
                                mx = fmaxf(mx, h_in[(ii * W + jj) * D + ll]);
                        }
                h_ref[(oi * OW + oj) * OD + ok] = mx;
            }
#endif

    int bad = test::compare(POOL_NAME, h_out, h_ref, out_n, 0.0f, 0.0f);
    int rc = test::report(POOL_NAME, bad, out_n);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
