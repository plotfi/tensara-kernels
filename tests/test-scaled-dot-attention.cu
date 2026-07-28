// Correctness test for scaled dot-product attention (no mask).
// Q,K,V are [B,H,S,E]; for each (b,h):
//   scores = Q @ K^T / sqrt(E)   (S x S)
//   P      = softmax(scores, over the last dim)
//   out    = P @ V               (S x E)

#include "test_utils.cuh"
extern "C" void solution(const float* Q, const float* K, const float* V, float* output,
                         size_t B, size_t H, size_t S, size_t E);

int main() {
    test::seed();
    size_t B = 2, H = 4, S = 32, E = 64;
    size_t total = B * H * S * E;

    float* h_q = new float[total];
    float* h_k = new float[total];
    float* h_v = new float[total];
    float* h_o = new float[total];
    float* h_ref = new float[total];
    test::fill_random(h_q, total);
    test::fill_random(h_k, total);
    test::fill_random(h_v, total);

    test::DBuf<float> d_q(total), d_k(total), d_v(total), d_o(total);
    d_q.upload(h_q); d_k.upload(h_k); d_v.upload(h_v);
    solution(d_q, d_k, d_v, d_o, B, H, S, E);
    test::check_cuda("scaled-dot-attention");
    d_o.download(h_o);

    double scale = 1.0 / sqrt((double)E);
    double* scores = new double[S];
    for (size_t b = 0; b < B; b++)
        for (size_t h = 0; h < H; h++) {
            const float* Qh = h_q + ((b * H + h) * S) * E;
            const float* Kh = h_k + ((b * H + h) * S) * E;
            const float* Vh = h_v + ((b * H + h) * S) * E;
            float* Oh = h_ref + ((b * H + h) * S) * E;
            for (size_t s = 0; s < S; s++) {
                double mx = -INFINITY;
                for (size_t t = 0; t < S; t++) {
                    double dot = 0.0;
                    for (size_t e = 0; e < E; e++) dot += (double)Qh[s * E + e] * Kh[t * E + e];
                    scores[t] = dot * scale;
                    if (scores[t] > mx) mx = scores[t];
                }
                double sum = 0.0;
                for (size_t t = 0; t < S; t++) { scores[t] = exp(scores[t] - mx); sum += scores[t]; }
                for (size_t e = 0; e < E; e++) {
                    double acc = 0.0;
                    for (size_t t = 0; t < S; t++) acc += scores[t] * Vh[t * E + e];
                    Oh[s * E + e] = static_cast<float>(acc / sum);
                }
            }
        }
    delete[] scores;

    int bad = test::compare("scaled-dot-attention", h_o, h_ref, total, 1e-3f, 1e-4f);
    int rc = test::report("scaled-dot-attention", bad, total);
    delete[] h_q; delete[] h_k; delete[] h_v; delete[] h_o; delete[] h_ref;
    return rc;
}
