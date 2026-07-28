// Correctness test for box-blur: KxK averaging filter.
// ASSUMED semantics: centered window, zero padding at borders, divide by the full
// kernel area (kernel_size^2, i.e. count_include_pad=True). kernel_size=3.
// If your problem clamps/reflects borders or divides by the valid-pixel count,
// adjust the reference below.

#include "test_utils.cuh"
extern "C" void solution(const float* input_image, int kernel_size,
                         float* output_image, size_t height, size_t width);

int main() {
    test::seed();
    size_t H = 64, W = 64, n = H * W;
    int ks = 3;

    float* h_in  = new float[n];
    float* h_out = new float[n];
    float* h_ref = new float[n];
    test::fill_random(h_in, n);

    test::DBuf<float> d_in(n), d_out(n);
    d_in.upload(h_in);
    solution(d_in, ks, d_out, H, W);
    test::check_cuda("box-blur");
    d_out.download(h_out);

    long r = (ks - 1) / 2;
    for (long i = 0; i < (long)H; i++)
        for (long j = 0; j < (long)W; j++) {
            double acc = 0.0;
            for (long di = -r; di <= r; di++)
                for (long dj = -r; dj <= r; dj++) {
                    long ii = i + di, jj = j + dj;
                    if (ii < 0 || ii >= (long)H || jj < 0 || jj >= (long)W) continue;
                    acc += h_in[ii * W + jj];
                }
            h_ref[i * W + j] = static_cast<float>(acc / (ks * ks));
        }

    int bad = test::compare("box-blur", h_out, h_ref, n, 1e-4f, 1e-5f);
    int rc = test::report("box-blur", bad, n);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
