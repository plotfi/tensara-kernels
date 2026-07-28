// Correctness test for conv-square-3d: "same"-size 3D cross-correlation with a
// cubic K^3 kernel, centered, zero padding. Cube side = size.
// C[x,y,z] = sum A[x+dx-r, y+dy-r, z+dz-r] * B[dx,dy,dz], r=(K-1)/2.

#include "test_utils.cuh"
extern "C" void solution(const float* A, const float* B, float* C, size_t size, size_t K);

int main() {
    test::seed();
    size_t S = 16, K = 3;

    float* h_a = new float[S * S * S];
    float* h_b = new float[K * K * K];
    float* h_c = new float[S * S * S];
    float* h_ref = new float[S * S * S];
    test::fill_random(h_a, S * S * S);
    test::fill_random(h_b, K * K * K);

    test::DBuf<float> d_a(S * S * S), d_b(K * K * K), d_c(S * S * S);
    d_a.upload(h_a); d_b.upload(h_b);
    solution(d_a, d_b, d_c, S, K);
    test::check_cuda("conv-square-3d");
    d_c.download(h_c);

    long r = (K - 1) / 2, Sl = S;
    auto A = [&](long x, long y, long z) { return h_a[(x * Sl + y) * Sl + z]; };
    for (long x = 0; x < Sl; x++)
        for (long y = 0; y < Sl; y++)
            for (long z = 0; z < Sl; z++) {
                double acc = 0.0;
                for (long dx = 0; dx < (long)K; dx++)
                    for (long dy = 0; dy < (long)K; dy++)
                        for (long dz = 0; dz < (long)K; dz++) {
                            long xx = x + dx - r, yy = y + dy - r, zz = z + dz - r;
                            if (xx < 0 || xx >= Sl || yy < 0 || yy >= Sl || zz < 0 || zz >= Sl) continue;
                            acc += static_cast<double>(A(xx, yy, zz)) * h_b[(dx * K + dy) * K + dz];
                        }
                h_ref[(x * Sl + y) * Sl + z] = static_cast<float>(acc);
            }

    int bad = test::compare("conv-square-3d", h_c, h_ref, S * S * S, 1e-4f, 1e-5f);
    int rc = test::report("conv-square-3d", bad, S * S * S);
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_ref;
    return rc;
}
