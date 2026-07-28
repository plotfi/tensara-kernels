// Correctness test for edge-detect (Sobel gradient magnitude).
// ASSUMED semantics: 3x3 Sobel operators, zero-padded borders,
//   Gx = [[-1,0,1],[-2,0,2],[-1,0,1]], Gy = Gx^T,
//   out = sqrt(gx^2 + gy^2).
// If your problem uses a different operator or normalization, adjust below.

#include "test_utils.cuh"
extern "C" void solution(const float* input_image, float* output_image,
                         size_t height, size_t width);

int main() {
    test::seed();
    size_t H = 64, W = 64, n = H * W;

    float* h_in  = new float[n];
    float* h_out = new float[n];
    float* h_ref = new float[n];
    test::fill_random(h_in, n, 0.0f, 1.0f);

    test::DBuf<float> d_in(n), d_out(n);
    d_in.upload(h_in);
    solution(d_in, d_out, H, W);
    test::check_cuda("edge-detect");
    d_out.download(h_out);

    const int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (long i = 0; i < (long)H; i++)
        for (long j = 0; j < (long)W; j++) {
            double sx = 0.0, sy = 0.0;
            for (int di = -1; di <= 1; di++)
                for (int dj = -1; dj <= 1; dj++) {
                    long ii = i + di, jj = j + dj;
                    if (ii < 0 || ii >= (long)H || jj < 0 || jj >= (long)W) continue;
                    double v = h_in[ii * W + jj];
                    sx += v * gx[di + 1][dj + 1];
                    sy += v * gy[di + 1][dj + 1];
                }
            h_ref[i * W + j] = static_cast<float>(sqrt(sx * sx + sy * sy));
        }

    int bad = test::compare("edge-detect", h_out, h_ref, n, 1e-4f, 1e-5f);
    int rc = test::report("edge-detect", bad, n);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
