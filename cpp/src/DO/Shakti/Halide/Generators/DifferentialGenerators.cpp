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

#include <DO/Shakti/Halide/Components/Differential.hpp>


namespace {

  using namespace Halide;

  template <typename T>
  class PolarGradient2D : public Generator<PolarGradient2D<T>>
  {
  public:
    using Base = Generator<PolarGradient2D<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<T>> input{"input", 2};
    Output<Buffer<>> output{"output", {Float(32), Float(32)}, 2};

    // TODO: extend to batches.
    Var x{"x"}, y{"y"};      // , n{"n"};
    Var xo{"xo"}, yo{"yo"};  // , no{"no"};
    Var xi{"xi"}, yi{"yi"};  // , ni{"ni"};

    void generate()
    {
      using DO::Shakti::HalideBackend::gradient;
      using DO::Shakti::HalideBackend::norm;

      auto input_padded = Halide::BoundaryConditions::repeat_edge(input);
      auto g = gradient(input_padded, x, y);
      auto mag = norm(g);
      auto ori = Halide::atan2(g(1), g(0));
      output(x, y) = {mag, ori};
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        output.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
                        TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Halide::Target::HVX_v62,  //
                                             Halide::Target::HVX_v65,
                                             Halide::Target::HVX_v66,
                                             Halide::Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.hexagon()
            .split(y, yo, yi, 128, TailStrategy::GuardWithIf)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        output.split(y, yo, yi, 8, TailStrategy::GuardWithIf)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);
      }
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(PolarGradient2D<float>,
                          shakti_polar_gradient_2d_32f_gpu)
