#pragma once

#include <DO/Shakti/Halide/MyHalide.hpp>
#include <DO/Shakti/Halide/Components/GaussianKernel.hpp>
#include <DO/Shakti/Halide/Components/SeparableConvolution.hpp>


namespace DO::Shakti::HalideBackend {

  template <typename Output = Halide::Func>
  struct GaussianConvolution2D {
    GaussianKernel gaussian;
    SeparableConvolution separable_conv_2d;

    int32_t tile_x{16};
    int32_t tile_y{16};

    Output output;

    template <typename Input>
    void generate(const Input& input, float sigma, int32_t truncation_factor, int w, int h)
    {
      gaussian.generate(sigma, truncation_factor);
      separable_conv_2d.generate(
          input,
          gaussian.kernel, gaussian.kernel_size, gaussian.kernel_shift,
          gaussian.kernel, gaussian.kernel_size, gaussian.kernel_shift,
          output,
          w, h);
    }

    void schedule(const Halide::Target & target)
    {
      gaussian.schedule();
      separable_conv_2d.schedule(target, tile_x, tile_y, output);
    }

    template <typename Input>
    void generate_2d(const Input& input, float sigma, int32_t truncation_factor, int w, int h)
    {
      gaussian.generate(sigma, truncation_factor);
      separable_conv_2d.generate_2d(
          input,
          gaussian.kernel, gaussian.kernel_size, gaussian.kernel_shift,
          gaussian.kernel, gaussian.kernel_size, gaussian.kernel_shift,
          output,
          w, h);
    }

    void schedule_2d(const Halide::Target & target)
    {
      gaussian.schedule();
      separable_conv_2d.schedule_2d(target, tile_x, tile_y, output);
    }

    inline operator Output&() {
      return output;
    }
  };

}  // namespace DO::Sara::HalideBackend
