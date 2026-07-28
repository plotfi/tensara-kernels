// Correctness test for triplet-margin loss (B triplets of dim E).
//   loss = mean_b max(0, ||a_b - p_b||_2 - ||a_b - n_b||_2 + margin)
// Euclidean (p=2) distances, mean over the batch. margin=1.0.

#include "test_utils.cuh"
extern "C" void solution(const float* anchor, const float* positive, const float* negative,
                         float* loss, size_t B, size_t E, float margin);

int main() {
    test::seed();
    size_t B = 8, E = 128;
    float margin = 1.0f;

    float* h_a = new float[B * E];
    float* h_p = new float[B * E];
    float* h_n = new float[B * E];
    float h_o = 0.0f, h_ref = 0.0f;
    test::fill_random(h_a, B * E);
    test::fill_random(h_p, B * E);
    test::fill_random(h_n, B * E);

    test::DBuf<float> d_a(B * E), d_p(B * E), d_n(B * E), d_o(1);
    d_a.upload(h_a); d_p.upload(h_p); d_n.upload(h_n);
    solution(d_a, d_p, d_n, d_o, B, E, margin);
    test::check_cuda("triplet-margin");
    d_o.download(&h_o);

    double total = 0.0;
    for (size_t b = 0; b < B; b++) {
        double dp = 0.0, dn = 0.0;
        for (size_t e = 0; e < E; e++) {
            double ap = h_a[b * E + e] - h_p[b * E + e];
            double an = h_a[b * E + e] - h_n[b * E + e];
            dp += ap * ap; dn += an * an;
        }
        double v = sqrt(dp) - sqrt(dn) + margin;
        total += v > 0.0 ? v : 0.0;
    }
    h_ref = static_cast<float>(total / B);

    int bad = test::compare("triplet-margin", &h_o, &h_ref, 1, 1e-3f, 1e-4f);
    int rc = test::report("triplet-margin", bad, 1);
    delete[] h_a; delete[] h_p; delete[] h_n;
    return rc;
}
