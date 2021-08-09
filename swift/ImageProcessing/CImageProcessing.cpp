#include "CImageProcessing.hpp"

#include <Halide.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "shakti_halide_rgb_to_gray.h"
#include "shakti_halide_gray32f_to_rgb.h"
#ifdef __cplusplus
}
#endif


void shakti_rgb8_to_gray32f_cpu(unsigned char* src, float* dst, int w, int h)
{
  auto src_buffer = Halide::Runtime::Buffer<unsigned char>::make_interleaved(src, w, h, 3);
  auto dst_buffer = Halide::Runtime::Buffer<float>{dst, w, h};
  shakti_halide_rgb_to_gray(src_buffer, dst_buffer);
}

void shakti_gray32f_to_rgb8_cpu(float* src, unsigned char* dst, int w, int h)
{
  auto src_buffer = Halide::Runtime::Buffer<float>{src, w, h};
  auto dst_buffer = Halide::Runtime::Buffer<unsigned char>::make_interleaved(dst, w, h, 3);
  shakti_halide_gray32f_to_rgb(src_buffer, dst_buffer);
}
