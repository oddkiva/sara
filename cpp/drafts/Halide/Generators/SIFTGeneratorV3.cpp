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

#include <drafts/Halide/Components/SIFT.hpp>


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

    Var k{"k"}, ko{"ko"}, ki{"ki"};
    Var ji{"ji"}, jio{"ji_outer"}, jii{"ji_inner"};
    Var o{"o"}, oo{"o_outer"}, oi{"o_inner"};

    DO::Shakti::HalideBackend::SIFT sift;

    Output<Buffer<float>> descriptors{"SIFT", 3};

    void generate()
    {
      const auto& mag = polar_gradient[0];
      const auto& ori = polar_gradient[1];
      const auto& mag_fn_ext = BoundaryConditions::constant_exterior(mag, 0);
      const auto& ori_fn_ext = BoundaryConditions::constant_exterior(ori, 0);

      const auto x = xyst[0](k);
      const auto y = xyst[1](k);
      const auto s = xyst[2](k);
      const auto theta = xyst[3](k);

      namespace halide = DO::Shakti::HalideBackend;
      // Initialize
      descriptors(o, ji, k) = 0.f;
      sift.accumulate_subhistogram(descriptors,                 //
                                   ji,                          //
                                   k,                           //
                                   mag_fn_ext, ori_fn_ext,      //
                                   x, y, s, scale_max, theta);  //
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        descriptors.gpu_tile(ji, k, jio, ko, tile_ji, tile_k,
                             Halide::TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        throw std::runtime_error{"Schedule not yet implemented for Hexagon!"};
      }

      // CPU schedule.
      else
      {
        // TODO: study
        // https://halide-lang.org/tutorials/tutorial_lesson_18_parallel_associative_reductions.html
        // And reapply this for dominant gradient orientations.
        descriptors.parallel(k);
      }
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(SIFTv3, shakti_sift_descriptor_v3)
