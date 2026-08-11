// Correctness test for cs4803dgc-lab3-conv2d-shared-constant (2D 5x5 convolution, 64x64) vs a CPU gold
// (matches CS4803VGCD/lab3/.../2Dconvolution_gold.cpp: border-clamped / zero-pad).
#include "test_utils.cuh"
extern "C" void solution(const float* filter, const float* N, float* P, size_t size);

int main() {
    test::seed();
    size_t sz = 64;
    float* hF = new float[25]; float* hN = new float[sz*sz];
    float* hP = new float[sz*sz]; float* hRef = new float[sz*sz];
    test::fill_random(hF, 25); test::fill_random(hN, sz*sz);

    test::DBuf<float> dF(25), dN(sz*sz), dP(sz*sz);
    dF.upload(hF); dN.upload(hN);
    solution(dF, dN, dP, sz);
    test::check_cuda("cs4803dgc-lab3-conv2d-shared-constant");
    dP.download(hP);

    for (size_t i = 0; i < sz; ++i)
        for (size_t j = 0; j < sz; ++j) {
            double sum = 0;
            unsigned mbegin = (i < 2) ? 2 - i : 0;
            unsigned mend   = (i > (sz - 3)) ? sz - i + 2 : 5;
            unsigned nbegin = (j < 2) ? 2 - j : 0;
            unsigned nend   = (j > (sz - 3)) ? (sz - j) + 2 : 5;
            for (unsigned m = mbegin; m < mend; ++m)
                for (unsigned n = nbegin; n < nend; n++)
                    sum += hF[m*5+n] * hN[sz*(i+m-2) + (j+n-2)];
            hRef[i*sz+j] = (float)sum;
        }

    int bad = test::compare("cs4803dgc-lab3-conv2d-shared-constant", hP, hRef, sz*sz, 1e-3f, 1e-4f);
    int rc = test::report("cs4803dgc-lab3-conv2d-shared-constant", bad, sz*sz);
    delete[] hF; delete[] hN; delete[] hP; delete[] hRef;
    return rc;
}
