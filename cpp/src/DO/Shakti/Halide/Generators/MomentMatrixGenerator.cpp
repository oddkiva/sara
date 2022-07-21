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

#include <DO/Shakti/Halide/MyHalide.hpp>


namespace v2 {

  using namespace Halide;

  class MomentMatrix : public Generator<MomentMatrix>
  {
  public:
    using Base = Generator<MomentMatrix>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<float>[2]> v { "v", 4 };
    Output<Buffer<>> moment{"moment", {Float(32), Float(32), Float(32)}, 4};

    //! 't' as time (since 'c' as channel does not make much sense).
    Var x{"x"}, y{"y"}, t{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, to{"co"};
    Var xi{"xi"}, yi{"yi"}, ti{"ci"};

    void generate()
    {
      auto fx = v[0](x, y, t, n);
      auto fy = v[1](x, y, t, n);

      auto fx_fx = fx * fx;
      auto fy_fy = fy * fy;
      auto fx_fy = fx * fy;
      moment(x, y, t, n) = {fx_fx, fy_fy, fx_fy};
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        moment.gpu_tile(x, y, t, xo, yo, to, xi, yi, ti, tile_x, tile_y, 1,
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

        moment.hexagon()
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        moment.split(y, yo, yi, 8, TailStrategy::GuardWithIf)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);
      }
    }
  };

  class Cornerness : public Generator<Cornerness>
  {
  public:
    using Base = Generator<Cornerness>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<float>[3]> moment { "moment", 4 };
    Input<float> kappa{"kappa"};
    Output<Buffer<float>> cornerness{"cornerness", 4};

    //! 't' as time (since 'c' as channel does not make much sense).
    Var x{"x"}, y{"y"}, t{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, to{"co"};
    Var xi{"xi"}, yi{"yi"}, ti{"ci"};

    void generate()
    {
      const auto mxx = moment[0](x, y, t, n);
      const auto myy = moment[1](x, y, t, n);
      const auto mxy = moment[2](x, y, t, n);

      const auto det = mxx * myy - mxy * mxy;
      const auto trace = mxx + myy;

      cornerness(x, y, t, n) = det - kappa * (trace * trace);
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        cornerness.gpu_tile(x, y, t, xo, yo, to, xi, yi, ti, tile_x, tile_y, 1,
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

        cornerness.hexagon()
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        cornerness.split(y, yo, yi, 8, TailStrategy::GuardWithIf)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);
      }
    }
  };

}  // namespace v2

HALIDE_REGISTER_GENERATOR(v2::MomentMatrix, shakti_moment_matrix_32f_cpu)
HALIDE_REGISTER_GENERATOR(v2::MomentMatrix, shakti_moment_matrix_32f_gpu)
HALIDE_REGISTER_GENERATOR(v2::MomentMatrix, shakti_cornerness_32f_cpu)
HALIDE_REGISTER_GENERATOR(v2::MomentMatrix, shakti_cornerness_32f_gpu)
