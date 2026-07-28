// Solution for "selu" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_SELU (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_SELU
#define ACT_SELU
#endif
#include "../kernel-implementation/activation.cuh"
