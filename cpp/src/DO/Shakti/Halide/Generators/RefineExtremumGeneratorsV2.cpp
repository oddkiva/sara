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

#include <DO/Shakti/Halide/Components/DoGExtremum.hpp>


namespace v2 {

  using namespace Halide;

  class RefineScaleSpaceExtrema
    : public Halide::Generator<RefineScaleSpaceExtrema>
  {
  public:
    GeneratorParam<int> tile_i{"tile_i", 32};

    // Scale-space function f(x, y, s)
    Input<Buffer<float>[3]> f{"f", 4};

    // Extrema.
    Input<Buffer<int32_t>[2]> extrema_xy{"xy", 1};
    Input<int32_t[2]> image_sizes{"image_sizes"};
    Input<float> scale{"scale"};
    Input<float> scale_factor{"scale_factor"};

    // Residuals.
    Output<Buffer<>> extrema_final{
        "extrema",                                     //
        {Float(32), Float(32), Float(32), Float(32)},  //
        1                                              //
    };                                                 //

    //! @brief Variables.
    //! @{
    Var i{"i"};
    Var io{"io"};
    Var ii{"ii"};
    //! @}

    void generate()
    {
      const auto& f_prev = Halide::BoundaryConditions::repeat_edge(f[0]);
      const auto& f_curr = Halide::BoundaryConditions::repeat_edge(f[1]);
      const auto& f_next = Halide::BoundaryConditions::repeat_edge(f[2]);

      const auto& x = extrema_xy[0];
      const auto& y = extrema_xy[1];

      const auto& w = image_sizes[0];
      const auto& h = image_sizes[1];

      const auto xi = Halide::clamp(x(i), 0, w);
      const auto yi = Halide::clamp(y(i), 0, h);
      const Expr c = 0;
      const Expr n = 0;

      using DO::Shakti::HalideBackend::refine_extremum_v1_4d;
      const auto result =
          refine_extremum_v1_4d(f_prev, f_curr, f_next, xi, yi, c, n);
          // refine_extremum_v2_4d(f_prev, f_curr, f_next, xi, yi, c, n);

      const auto& dx = result[0];
      const auto& dy = result[1];
      const auto& ds = result[2];
      const auto& val = result[3];
      const auto& success = result[4];

      const auto xf = cast<float>(xi);
      const auto yf = cast<float>(yi);

      const auto x_final = select(success, xf + dx, xf);
      const auto y_final = select(success, yf + dy, yf);
      const auto s_final = select(success,                        //
                                  scale * pow(scale_factor, ds),  //
                                  scale);                         //
      const auto val_final = select(success, val, f_curr(xi, yi, c, n));

      extrema_final(i) = {x_final, y_final, s_final, val_final};
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        extrema_final.gpu_tile(i, io, ii, tile_i, TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Halide::Target::HVX_v62,  //
                                             Halide::Target::HVX_v65,
                                             Halide::Target::HVX_v66,
                                             Halide::Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        extrema_final.hexagon()
            .split(i, io, ii, 128)
            .parallel(io)
            .vectorize(i, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        extrema_final.split(i, io, ii, 8)
            .parallel(io)
            .vectorize(i, 8, TailStrategy::GuardWithIf);
      }
    }
  };

}  // namespace

namespace v3 {

  using namespace Halide;

  class RefineScaleSpaceExtrema
    : public Halide::Generator<RefineScaleSpaceExtrema>
  {
  public:
    GeneratorParam<int> tile_i{"tile_i", 32};

    // Scale-space function f(x, y, s, n)
    Input<Buffer<float>> f{"f", 4};

    // Extrema.
    Input<Buffer<std::int32_t>[4]> extrema_xysn{"xysn", 1};
    Input<Buffer<float>> scale{"scale", 1};
    Input<float> scale_factor{"scale_factor"};

    // Residuals.
    Output<Buffer<>> extrema_final{
        "extrema",                                     //
        {Float(32), Float(32), Float(32), Float(32)},  //
        1                                              //
    };                                                 //

    //! @brief Variables.
    //! @{
    Var i{"i"};
    Var io{"io"};
    Var ii{"ii"};
    //! @}

    void generate()
    {
      auto f_ext = Halide::BoundaryConditions::repeat_edge(f);

      const auto& x = extrema_xysn[0](i);
      const auto& y = extrema_xysn[1](i);
      const auto& s = extrema_xysn[2](i);
      const auto& n = extrema_xysn[3](i);

      using DO::Shakti::HalideBackend::refine_extremum_v1_batch;
      const auto result = refine_extremum_v1_batch(f_ext, x, y, s, n);

      const auto& dx = result[0];
      const auto& dy = result[1];
      const auto& ds = result[2];
      const auto& val = result[3];
      const auto& success = result[4];

      const auto xf = cast<float>(x);
      const auto yf = cast<float>(y);

      const auto x_final = select(success, xf + dx, xf);
      const auto y_final = select(success, yf + dy, yf);
      const auto s_final = select(success,                        //
                                  scale(i) * pow(scale_factor, ds),  //
                                  scale(i));                         //
      const auto val_final = select(success, val, f_ext(x, y, s, n));

      extrema_final(i) = {x_final, y_final, s_final, val_final};
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        extrema_final.gpu_tile(i, io, ii, tile_i, TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Halide::Target::HVX_v62,  //
                                             Halide::Target::HVX_v65,
                                             Halide::Target::HVX_v66,
                                             Halide::Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        extrema_final.hexagon()
            .split(i, io, ii, 128)
            .parallel(io)
            .vectorize(i, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        extrema_final.split(i, io, ii, 8)
            .parallel(io)
            .vectorize(i, 8, TailStrategy::GuardWithIf);
      }
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(v2::RefineScaleSpaceExtrema,
                          shakti_refine_scale_space_extrema_gpu_v2)
HALIDE_REGISTER_GENERATOR(v3::RefineScaleSpaceExtrema,
                          shakti_refine_scale_space_extrema_gpu_v3)
