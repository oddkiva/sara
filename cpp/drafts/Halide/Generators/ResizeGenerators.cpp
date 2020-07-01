// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <drafts/Halide/MyHalide.hpp>
#include <drafts/Halide/Components/GaussianKernel.hpp>
#include <drafts/Halide/Components/ScaleComponent.hpp>
#include <drafts/Halide/Components/SeparableConvolution.hpp>


namespace {

  using namespace Halide;

  template <typename T>
  class Scale : public Generator<Scale<T>>
  {
  public:
    using Base = Generator<Scale<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<T>> input{"input", 4};
    Input<int32_t[2]> output_sizes{"output_sizes"};
    Output<Buffer<T>> output{"output", 4};

    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"}, no{"no"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"}, ni{"ni"};

    void generate()
    {
      Expr w_in = cast<float>(input.dim(0).extent());
      Expr h_in = cast<float>(input.dim(1).extent());
      Expr w_out = cast<float>(output_sizes[0]);
      Expr h_out = cast<float>(output_sizes[1]);

      Expr xx = cast<int32_t>(x * w_in / w_out);
      Expr yy = cast<int32_t>(y * h_in / h_out);

      output(x, y, c, n) = input(xx, yy, c, n);
    }

    void schedule()
    {
      input.dim(0).set_stride(Expr());  // Use an undefined Expr to
                                        // mean there is no
                                        // constraint.

      output.dim(0).set_stride(Expr());

      Expr input_is_planar = input.dim(0).stride() == 1;
      Expr input_is_interleaved = input.dim(0).stride() == 3 &&  //
                                  input.dim(2).stride() == 1 &&  //
                                  input.dim(2).extent() == 3;

      Expr output_is_planar = output.dim(0).stride() == 1;
      Expr output_is_interleaved = output.dim(0).stride() == 3 &&  //
                                   output.dim(2).stride() == 1 &&  //
                                   output.dim(2).extent() == 3;

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        output.specialize(input_is_planar && output_is_planar)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                      TailStrategy::GuardWithIf);

        output.specialize(input_is_interleaved && output_is_interleaved)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 3,
                      TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.specialize(input_is_planar && output_is_planar)
            .hexagon()
            .prefetch(input, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        output.specialize(input_is_planar && output_is_planar)
            .split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);

        output.specialize(input_is_interleaved && output_is_interleaved)
            .reorder(c, x, y)
            .unroll(c);
      }
    }
  };

  template <typename T>
  class Reduce : public Generator<Reduce<T>>
  {
  public:
    using Base = Generator<Reduce<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};
    GeneratorParam<int32_t> truncation_factor{"truncation_factor", 4};

    Input<Buffer<T>> input{"input", 4};
    Input<int32_t[2]> output_sizes{"output_sizes"};

    // Gaussian convolution component.
    DO::Shakti::HalideBackend::GaussianKernel gx;
    DO::Shakti::HalideBackend::GaussianKernel gy;

    DO::Shakti::HalideBackend::SeparableConvolution separable_conv_2d;
    Func input_blurred{"input_blurred"};

    // Downscale component.
    ScaleComponent downscale;

    // The realization.
    Output<Buffer<T>> output{"input_reduced", 4};

    void generate()
    {
      const auto w_in = input.dim(0).extent();
      const auto h_in = input.dim(1).extent();
      const auto w_out = output_sizes[0];
      const auto h_out = output_sizes[1];

      const auto wi = cast<float>(w_in);
      const auto hi = cast<float>(h_in);
      const auto wo = cast<float>(w_out);
      const auto ho = cast<float>(h_out);

      const auto sx = wi / wo;
      const auto sy = hi / ho;

      const auto sigma_x = 1.6f * sqrt(sx * sx - 0.99f);
      gx.generate(sigma_x, truncation_factor);

      const auto sigma_y = 1.6f * sqrt(sy * sy - 0.99f);
      gy.generate(sigma_y, truncation_factor);

      separable_conv_2d.generate(
          input,
          gx.kernel, gx.kernel_size, gx.kernel_shift,
          gy.kernel, gy.kernel_size, gy.kernel_shift,
          input_blurred, w_in, h_in);

      downscale.generate(input_blurred, output, w_in, h_in, w_out, h_out);
    }

    void schedule()
    {
      gx.schedule();
      gy.schedule();

      separable_conv_2d.schedule(get_target(), tile_x, tile_y, input_blurred);
      input_blurred.compute_root();

      downscale.schedule(get_target(), tile_x, tile_y, input_blurred, output);
    }
  };

  class Enlarge : public Halide::Generator<Enlarge>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 8};
    GeneratorParam<int> tile_y{"tile_y", 8};

    Input<Buffer<float>> input{"input", 3};
    Input<int[2]> in_sizes{"in_sizes"};
    Input<int[2]> out_sizes{"out_sizes"};
    Output<Buffer<float>> output{"enlarged_input", 3};

    Var x{"x"}, y{"y"}, c{"c"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};

    void generate()
    {
      Expr w_in = cast<float>(in_sizes[0]);
      Expr h_in = cast<float>(in_sizes[1]);
      Expr w_out = cast<float>(out_sizes[0]);
      Expr h_out = cast<float>(out_sizes[1]);

      Expr xx = x * w_in / w_out;
      Expr yy = y * h_in / h_out;

      auto input_padded = BoundaryConditions::repeat_edge(input);

      auto wx = xx - floor(xx);
      auto wy = yy - floor(yy);

      auto xr = cast<int>(xx);
      auto yr = cast<int>(yy);

      output(x, y, c) = (1 - wx) * (1 - wy) * input_padded(xr, yr, c) +  //
                        wx * (1 - wy) * input_padded(xr + 1, yr, c) +    //
                        (1 - wx) * wy * input_padded(xr, yr + 1, c) +    //
                        wx * wy * input_padded(xr + 1, yr + 1, c);
    }

    void schedule()
    {
      input.dim(0).set_stride(Expr());  // Use an undefined Expr to
                                        // mean there is no
                                        // constraint.

      output.dim(0).set_stride(Expr());

      Expr input_is_planar = input.dim(0).stride() == 1;
      Expr input_is_interleaved = input.dim(0).stride() == 3 &&  //
                                  input.dim(2).stride() == 1 &&  //
                                  input.dim(2).extent() == 3;

      Expr output_is_planar = output.dim(0).stride() == 1;
      Expr output_is_interleaved = output.dim(0).stride() == 3 &&  //
                                   output.dim(2).stride() == 1 &&  //
                                   output.dim(2).extent() == 3;

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        output.specialize(input_is_planar && output_is_planar)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                      TailStrategy::GuardWithIf);

        output.specialize(input_is_interleaved && output_is_interleaved)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 3,
                      TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.specialize(input_is_planar && output_is_planar)
            .hexagon()
            .prefetch(input, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        output.specialize(input_is_planar && output_is_planar)
            .split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);

        output.specialize(input_is_interleaved && output_is_interleaved)
            .reorder(c, x, y)
            .unroll(c);
      }
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(Scale<float>, shakti_scale_32f)
HALIDE_REGISTER_GENERATOR(Reduce<float>, shakti_reduce_32f)
HALIDE_REGISTER_GENERATOR(Enlarge, shakti_enlarge)
