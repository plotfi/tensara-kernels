// Correctness test for conv-2d: "same"-size 2D cross-correlation, centered
// kernel, zero padding (matching the repo's conv-1d convention).
// C[i,j] = sum_{di,dj} A[i+di-rh, j+dj-rw] * B[di,dj], rh=(Kh-1)/2, rw=(Kw-1)/2.

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, float* C,
                         size_t H, size_t W, size_t Kh, size_t Kw);

int main() {
    test::seed();
    size_t H = 64, W = 64, Kh = 3, Kw = 3;

    float* h_a = new float[H * W];
    float* h_b = new float[Kh * Kw];
    float* h_c = new float[H * W];
    float* h_ref = new float[H * W];
    test::fill_random(h_a, H * W);
    test::fill_random(h_b, Kh * Kw);

    test::DBuf<float> d_a(H * W), d_b(Kh * Kw), d_c(H * W);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_c, H, W, Kh, Kw);
    test::check_cuda("conv-2d");
    d_c.download(h_c);

    long rh = (Kh - 1) / 2, rw = (Kw - 1) / 2;
    for (long i = 0; i < (long)H; i++)
        for (long j = 0; j < (long)W; j++) {
            double acc = 0.0;
            for (long di = 0; di < (long)Kh; di++)
                for (long dj = 0; dj < (long)Kw; dj++) {
                    long ii = i + di - rh, jj = j + dj - rw;
                    if (ii < 0 || ii >= (long)H || jj < 0 || jj >= (long)W) continue;
                    acc += static_cast<double>(h_a[ii * W + jj]) * h_b[di * Kw + dj];
                }
            h_ref[i * W + j] = static_cast<float>(acc);
        }

    int bad = test::compare("conv-2d", h_c, h_ref, H * W, 1e-4f, 1e-5f);
    int rc = test::report("conv-2d", bad, H * W);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
