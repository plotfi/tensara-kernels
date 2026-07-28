// Correctness test for conv2d-relu-hardswish: conv2d ("same", centered, zero-pad)
// followed by ReLU then HardSwish.
//   t = conv2d(image, kernel);  y = hardswish(relu(t))
//   hardswish(x) = x * clamp(x + 3, 0, 6) / 6

#include "test_utils.cuh"
extern "C" void solution(const float* image, const float* kernel, float* output,
                         size_t H, size_t W, size_t Kh, size_t Kw);

int main() {
    test::seed();
    size_t H = 64, W = 64, Kh = 3, Kw = 3;

    float* h_img = new float[H * W];
    float* h_ker = new float[Kh * Kw];
    float* h_out = new float[H * W];
    float* h_ref = new float[H * W];
    test::fill_random(h_img, H * W);
    test::fill_random(h_ker, Kh * Kw);

    test::DBuf<float> d_img(H * W), d_ker(Kh * Kw), d_out(H * W);
    d_img.upload(h_img); d_ker.upload(h_ker);
    solution(d_img, d_ker, d_out, H, W, Kh, Kw);
    test::check_cuda("conv2d-relu-hardswish");
    d_out.download(h_out);

    long rh = (Kh - 1) / 2, rw = (Kw - 1) / 2;
    for (long i = 0; i < (long)H; i++)
        for (long j = 0; j < (long)W; j++) {
            double acc = 0.0;
            for (long di = 0; di < (long)Kh; di++)
                for (long dj = 0; dj < (long)Kw; dj++) {
                    long ii = i + di - rh, jj = j + dj - rw;
                    if (ii < 0 || ii >= (long)H || jj < 0 || jj >= (long)W) continue;
                    acc += static_cast<double>(h_img[ii * W + jj]) * h_ker[di * Kw + dj];
                }
            double r = acc > 0.0 ? acc : 0.0;          // relu
            double clamped = r + 3.0;                   // hardswish
            clamped = clamped < 0.0 ? 0.0 : (clamped > 6.0 ? 6.0 : clamped);
            h_ref[i * W + j] = static_cast<float>(r * clamped / 6.0);
        }

    int bad = test::compare("conv2d-relu-hardswish", h_out, h_ref, H * W, 1e-4f, 1e-5f);
    int rc = test::report("conv2d-relu-hardswish", bad, H * W);
    delete[] h_img; delete[] h_ker; delete[] h_out; delete[] h_ref;
    return rc;
}
