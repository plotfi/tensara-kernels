// Correctness test for threshold (binary thresholding).
// ASSUMED semantics: out = (in > threshold_value) ? 1.0 : 0.0.
// If your problem keeps the value instead (in > t ? in : 0), adjust the reference.

#include "test_utils.cuh"
extern "C" void solution(const float* input_image, float threshold_value,
                         float* output_image, size_t height, size_t width);

int main() {
    test::seed();
    size_t H = 64, W = 64, n = H * W;
    float thr = 0.5f;

    float* h_in  = new float[n];
    float* h_out = new float[n];
    float* h_ref = new float[n];
    test::fill_random(h_in, n, 0.0f, 1.0f);

    test::DBuf<float> d_in(n), d_out(n);
    d_in.upload(h_in);
    solution(d_in, thr, d_out, H, W);
    test::check_cuda("threshold");
    d_out.download(h_out);

    for (size_t i = 0; i < n; i++) h_ref[i] = (h_in[i] > thr) ? 1.0f : 0.0f;

    int bad = test::compare("threshold", h_out, h_ref, n, 0.0f, 0.0f);
    int rc = test::report("threshold", bad, n);
    delete[] h_in; delete[] h_out; delete[] h_ref;
    return rc;
}
