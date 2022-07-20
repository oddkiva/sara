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

#include <DO/Shakti/Halide/Components/TinyLinearAlgebra.hpp>


namespace v2 {

  using namespace Halide;

  class CartesianToPolarCoordinates
    : public Generator<CartesianToPolarCoordinates>
  {
  public:
    using Base = Generator<CartesianToPolarCoordinates>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<float>[2]> cart { "cartesian", 4 };
    Output<Buffer<>> polar{"polar", {Float(32), Float(32)}, 4};

    //! 't' as time (since 'c' as channel does not make much sense).
    Var x{"x"}, y{"y"}, t{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, to{"co"};
    Var xi{"xi"}, yi{"yi"}, ti{"ci"};

    void generate()
    {
      using DO::Shakti::HalideBackend::norm;

      auto fx = cart[0](x, y, t, n);
      auto fy = cart[1](x, y, t, n);
      auto mag = Halide::sqrt(fx * fx + fy * fy);
#define USE_FAST_ATAN2
#ifdef USE_FAST_ATAN2
      // See njuffa's answer where he exposes Remez's polynomial approximation.
      // He reports that the relative error is 1/360, which is more than
      // acceptable for image processing applications.
      //
      // https://math.stackexchange.com/questions/1098487/atan2-faster-approximation/1105038
      //
      // We translate his pseudo-code as-is.
      const auto fast_atan2 = [](const Expr& y, const Expr& x) -> Expr {
        const auto abs_x = abs(x);
        const auto abs_y = abs(y);
        const auto max_x_y = max(abs_x, abs_y);

        // Here is a boundary condition we need to take care to avoid NaN values
        // which the post did not address.
        const auto a = select(max_x_y != 0,  //
                              min(abs_x, abs_y) / max_x_y, 0.f);

        // Calculate the 7-th order polynomial using Horner's method.
        const auto s = a * a;
        auto r =
            ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
        r = select(abs_y > abs_x, 1.57079637f - r, r);
        r = select(x < 0, 3.14159274f - r, r);
        r = select(y < 0, -r, r);

        return r;
      };
      const auto ori = fast_atan2(fy, fx);
#else
      auto ori = Halide::atan2(fy, fx);
#endif
      polar(x, y, t, n) = {mag, ori};
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        polar.gpu_tile(x, y, t, xo, yo, to, xi, yi, ti, tile_x, tile_y, 1,
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

        polar.hexagon()
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        polar.split(y, yo, yi, 8, TailStrategy::GuardWithIf)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);
      }
    }
  };

}  // namespace v2

HALIDE_REGISTER_GENERATOR(v2::CartesianToPolarCoordinates,
                          shakti_cartesian_to_polar_32f_cpu)
HALIDE_REGISTER_GENERATOR(v2::CartesianToPolarCoordinates,
                          shakti_cartesian_to_polar_32f_gpu)
