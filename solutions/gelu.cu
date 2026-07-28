// Solution for "gelu" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_GELU (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_GELU
#define ACT_GELU
#endif
#include "../kernel-implementation/activation.cuh"
