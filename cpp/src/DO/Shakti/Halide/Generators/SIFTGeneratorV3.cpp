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

#include <array>

#include <DO/Shakti/Halide/Components/SIFT.hpp>


namespace {

  using namespace Halide;

  class SIFTv3 : public Generator<SIFTv3>
  {
  public:
    GeneratorParam<int> tile_ji{"tile_ji", 16};
    GeneratorParam<int> tile_k{"tile_k", 32};

    // Input data.
    Input<Buffer<float>[2]> polar_gradient { "gradients", 2 };
    Input<Buffer<float>[4]> xyst { "xyst", 1 };
    Input<float> scale_max{"scale_max"};
    Input<std::int32_t> N{"N"};
    Input<std::int32_t> O{"O"};

    Var u{"u"}, v{"v"};
    Var k{"k"}, ko{"ko"}, ki{"ki"};
    Var ji{"ji"}, jio{"ji_outer"}, jii{"ji_inner"};
    Var o{"o"}, oo{"o_outer"}, oi{"o_inner"};

    DO::Shakti::HalideBackend::SIFT sift;
    Halide::Func gradient_patch{"gradient_patch"};

    Output<Buffer<float>> descriptors{"SIFT", 3};

    void generate()
    {
      const auto& mag = polar_gradient[0];
      const auto& ori = polar_gradient[1];
      const auto& mag_fn = BoundaryConditions::constant_exterior(mag, 0);
      const auto& ori_fn = BoundaryConditions::constant_exterior(ori, 0);

      const auto x = xyst[0](k);
      const auto y = xyst[1](k);
      const auto s = xyst[2](k);
      const auto theta = xyst[3](k);

      namespace halide = DO::Shakti::HalideBackend;

//#define TRY_V2
#ifdef TRY_V2
      // Retrieve the (i, j) coordinates of the corresponding image subpatch.
      const auto i = ji / N;
      const auto j = ji % N;

      const Halide::Expr bin_length_in_pixels =
          sift.bin_length_in_scale_unit * s;

      const auto dx_ij = cos(theta) * bin_length_in_pixels *
                         (Halide::cast<float>(j) - N / 2.f + 0.5f);
      const auto dy_ij = sin(theta) * bin_length_in_pixels *
                         (Halide::cast<float>(i) - N / 2.f + 0.5f);
      const auto x_ij = Halide::cast<std::int32_t>(Halide::round(x + dx_ij));
      const auto y_ij = Halide::cast<std::int32_t>(Halide::round(y + dy_ij));

      gradient_patch(u, v, ji, k) = Halide::Tuple{
          mag_fn(x_ij + u, y_ij + v),  //
          ori_fn(x_ij + u, y_ij + v)   //
      };

      descriptors(o, ji, k) = 0.f;
      sift.accumulate_subhistogram_v2(descriptors,                 //
                                      ji,                          //
                                      k,                           //
                                      gradient_patch,              //
                                      x, y, s, scale_max, theta);  //
#else
      // Initialize
      descriptors(o, ji, k) = 0.f;
      sift.accumulate_subhistogram(descriptors,                 //
                                   ji,                          //
                                   k,                           //
                                   mag_fn, ori_fn,              //
                                   x, y, s, scale_max, theta);  //
#endif
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
#ifdef TRY_V2
        gradient_patch.compute_root();
        gradient_patch.gpu_tile(ji, k, jio, ko, jii, ki, tile_ji, tile_k,
                             Halide::TailStrategy::GuardWithIf);
#endif
        descriptors.gpu_tile(ji, k, jio, ko, jii, ki, tile_ji, tile_k,
                             Halide::TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Halide::Target::HVX_v62,  //
                                             Halide::Target::HVX_v65,
                                             Halide::Target::HVX_v66,
                                             Halide::Target::HVX_128}))
      {
        throw std::runtime_error{"Schedule not yet implemented for Hexagon!"};
      }

      // CPU schedule.
      else
      {
        descriptors.parallel(k);//.split(ji, jio, jii, tile_ji).vectorize(o, 8);
      }
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(SIFTv3, shakti_sift_descriptor_gpu_v3)
