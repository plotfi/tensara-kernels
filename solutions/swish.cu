// Solution for "swish" (elementwise activation).
// Selects the op, then pulls in the shared vectorized activation template.
// Guarded so building it alongside a test compiled with -DACT_SWISH (for the test's
// CPU reference) doesn't trigger a macro-redefinition warning.
#ifndef ACT_SWISH
#define ACT_SWISH
#endif
#include "../kernel-implementation/activation.cuh"
