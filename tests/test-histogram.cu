// Correctness test for histogram.
// ASSUMED semantics: num_bins uniform bins spanning the fixed range [0,1); a
// value v maps to bin clamp(floor(v*num_bins), 0, num_bins-1); the histogram
// holds per-bin counts (as float). Inputs are drawn in [0,1).
// If your problem bins over the data's own [min,max] or a different range,
// adjust the reference.

#include "test_utils.cuh"
extern "C" void solution(const float* image, int num_bins, float* histogram,
                         size_t height, size_t width);

int main() {
    test::seed();
    size_t H = 64, W = 64, n = H * W;
    int num_bins = 256;

    float* h_img = new float[n];
    float* h_out = new float[num_bins];
    float* h_ref = new float[num_bins];
    test::fill_random(h_img, n, 0.0f, 0.999999f);
    for (int b = 0; b < num_bins; b++) h_ref[b] = 0.0f;

    test::DBuf<float> d_img(n), d_hist(num_bins);
    d_img.upload(h_img);
    solution(d_img, num_bins, d_hist, H, W);
    test::check_cuda("histogram");
    d_hist.download(h_out);

    for (size_t i = 0; i < n; i++) {
        int b = (int)floorf(h_img[i] * num_bins);
        if (b < 0) b = 0;
        if (b >= num_bins) b = num_bins - 1;
        h_ref[b] += 1.0f;
    }

    int bad = test::compare("histogram", h_out, h_ref, num_bins, 0.0f, 0.0f);
    int rc = test::report("histogram", bad, num_bins);
    delete[] h_img; delete[] h_out; delete[] h_ref;
    return rc;
}
