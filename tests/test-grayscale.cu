// Correctness test for grayscale: RGB -> luminance (Rec. 601).
// ASSUMED layout: interleaved HWC, pixel p at [3p, 3p+1, 3p+2] = (R,G,B).
//   gray[p] = 0.299*R + 0.587*G + 0.114*B

#include "test_utils.cuh"
extern "C" void solution(const float* rgb_image, float* grayscale_output,
                         size_t height, size_t width, size_t channels);

int main() {
    test::seed();
    size_t H = 64, W = 64, C = 3;
    size_t px = H * W;

    float* h_rgb  = new float[px * C];
    float* h_gray = new float[px];
    float* h_ref  = new float[px];
    test::fill_random(h_rgb, px * C, 0.0f, 1.0f); // image intensities in [0,1]

    test::DBuf<float> d_rgb(px * C), d_gray(px);
    d_rgb.upload(h_rgb);
    solution(d_rgb, d_gray, H, W, C);
    test::check_cuda("grayscale");
    d_gray.download(h_gray);

    for (size_t p = 0; p < px; p++)
        h_ref[p] = 0.299f * h_rgb[3 * p] + 0.587f * h_rgb[3 * p + 1] + 0.114f * h_rgb[3 * p + 2];

    int bad = test::compare("grayscale", h_gray, h_ref, px, 1e-4f, 1e-5f);
    int rc = test::report("grayscale", bad, px);
    delete[] h_rgb; delete[] h_gray; delete[] h_ref;
    return rc;
}
