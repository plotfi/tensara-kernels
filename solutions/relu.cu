// Solution for "relu" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_RELU (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_RELU
#define ACT_RELU
#endif
#include "../kernel-implementation/activation.cuh"
