// Correctness test for the scalar 1D losses (output[0], n elements). Selected at
// compile time:
//   -DL_HUBER -> huber-loss    (delta=1.0; per-elem 0.5e^2 if |e|<=delta else delta*(|e|-0.5*delta))
//   -DL_HINGE -> hinge-loss    (per-elem max(0, 1 - pred*tgt))
//   -DL_KL    -> kl-loss       (per-elem t*(log t - log p); inputs made positive)
// All use the MEAN reduction over n. If your problem uses sum/batchmean, adjust.

#include "test_utils.cuh"
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n);

#if defined(L_HINGE)
#  define LOSS_NAME "hinge-loss"
#elif defined(L_KL)
#  define LOSS_NAME "kl-loss"
#else
#  define LOSS_NAME "huber-loss"
#endif

int main() {
    test::seed();
    size_t n = 1024;

    float* h_p = new float[n];
    float* h_t = new float[n];
    float h_o = 0.0f, h_ref = 0.0f;
    test::fill_random(h_p, n);
    test::fill_random(h_t, n);
#if defined(L_KL)
    // KL needs positive p, t; use strictly-positive inputs in (0,1].
    test::fill_random(h_p, n, 1e-3f, 1.0f);
    test::fill_random(h_t, n, 1e-3f, 1.0f);
#endif

    test::DBuf<float> d_p(n), d_t(n), d_o(1);
    d_p.upload(h_p); d_t.upload(h_t);
    solution(d_p, d_t, d_o, n);
    test::check_cuda(LOSS_NAME);
    d_o.download(&h_o);

    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        double p = h_p[i], t = h_t[i];
#if defined(L_HINGE)
        double e = 1.0 - p * t; acc += e > 0.0 ? e : 0.0;
#elif defined(L_KL)
        acc += t * (log(t) - log(p));
#else // huber, delta = 1
        double e = fabs(p - t);
        acc += e <= 1.0 ? 0.5 * e * e : (e - 0.5);
#endif
    }
    h_ref = static_cast<float>(acc / n);

    int bad = test::compare(LOSS_NAME, &h_o, &h_ref, 1, 1e-3f, 1e-4f);
    int rc = test::report(LOSS_NAME, bad, 1);
    delete[] h_p; delete[] h_t;
    return rc;
}
