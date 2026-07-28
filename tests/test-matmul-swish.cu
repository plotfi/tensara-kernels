// Correctness test for matmul-swish: a linear layer with Swish and scaling.
// input[bs,in], weight[out,in], bias[out] -> output[bs,out]:
//   output = swish(input @ weight^T + bias) * scaling_factor
//   swish(x) = x * sigmoid(x)

#include "test_utils.cuh"
extern "C" void solution(const float* input_matrix, const float* weight_matrix,
                         const float* bias, float scaling_factor, float* output,
                         size_t batch_size, size_t in_features, size_t out_features);

int main() {
    test::seed();
    size_t bs = 8, in = 64, out = 32;
    float scaling = 1.0f;

    float* h_x = new float[bs * in];
    float* h_w = new float[out * in];
    float* h_b = new float[out];
    float* h_o = new float[bs * out];
    float* h_ref = new float[bs * out];
    test::fill_random(h_x, bs * in);
    test::fill_random(h_w, out * in);
    test::fill_random(h_b, out);

    test::DBuf<float> d_x(bs * in), d_w(out * in), d_b(out), d_o(bs * out);
    d_x.upload(h_x); d_w.upload(h_w); d_b.upload(h_b);
    solution(d_x, d_w, d_b, scaling, d_o, bs, in, out);
    test::check_cuda("matmul-swish");
    d_o.download(h_o);

    for (size_t i = 0; i < bs; i++)
        for (size_t j = 0; j < out; j++) {
            double acc = h_b[j];
            for (size_t p = 0; p < in; p++)
                acc += static_cast<double>(h_x[i * in + p]) * h_w[j * in + p];
            double sw = acc / (1.0 + exp(-acc));
            h_ref[i * out + j] = static_cast<float>(sw * scaling);
        }

    int bad = test::compare("matmul-swish", h_o, h_ref, bs * out, 1e-3f, 1e-3f);
    int rc = test::report("matmul-swish", bad, bs * out);
    delete[] h_x; delete[] h_w; delete[] h_b; delete[] h_o; delete[] h_ref;
    return rc;
}
