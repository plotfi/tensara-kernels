// Solution for "threshold": binary thresholding, out = (x > t) ? 1 : 0.
// Reuses the unified elementwise activation kernel -- threshold has the same
// (float x, float alpha) shape, with alpha carrying the threshold value.
#include "../kernel-implementation/activation.cuh"

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    multi_solution<threshold, 512>(input_image, output_image, height, width, threshold_value);
}
