// Correctness test for mse-loss: output[0] = mean over all elements of
// (predictions - targets)^2. Shape is [64,64]; reduction is the mean.

#include "test_utils.cuh"
extern "C" void solution(const float* predictions, const float* targets, float* output,
                         const size_t* shape, size_t ndim);

int main() {
    test::seed();
    size_t rows = 64, cols = 64, n = rows * cols;
    size_t ndim = 2;

    float* h_p = new float[n];
    float* h_t = new float[n];
    float h_o = 0.0f, h_ref = 0.0f;
    test::fill_random(h_p, n);
    test::fill_random(h_t, n);

    test::DBuf<float> d_p(n), d_t(n), d_o(1);
    test::DBuf<size_t> d_shape(ndim);
    d_p.upload(h_p); d_t.upload(h_t);
    size_t shape[2] = {rows, cols};
    d_shape.upload(shape);

    solution(d_p, d_t, d_o, d_shape, ndim);
    test::check_cuda("mse-loss");
    d_o.download(&h_o);

    double acc = 0.0;
    for (size_t i = 0; i < n; i++) { double e = (double)h_p[i] - h_t[i]; acc += e * e; }
    h_ref = static_cast<float>(acc / n);

    int bad = test::compare("mse-loss", &h_o, &h_ref, 1, 1e-4f, 1e-5f);
    int rc = test::report("mse-loss", bad, 1);
    delete[] h_p; delete[] h_t;
    return rc;
}
