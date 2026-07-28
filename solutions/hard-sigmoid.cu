// Solution for "hard-sigmoid" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_HARD_SIGMOID (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_HARD_SIGMOID
#define ACT_HARD_SIGMOID
#endif
#include "../kernel-implementation/activation.cuh"
