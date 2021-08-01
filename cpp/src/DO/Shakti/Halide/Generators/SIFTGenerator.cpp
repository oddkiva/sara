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

  class SIFT : public Generator<SIFT>
  {
  public:
    GeneratorParam<int> tile_i{"tile_i", 4};
    GeneratorParam<int> tile_j{"tile_j", 4};
    GeneratorParam<int> tile_o{"tile_o", 8};

    // Input data.
    Input<Buffer<float>[2]> polar_gradient { "gradients", 2 };
    Input<Buffer<float>[4]> xyst { "xyst", 1 };
    Input<float> scale_max{"scale_max"};
    Input<float> bin_length_in_scale_unit{"bin_length_in_scale_unit"};
    Input<std::int32_t> N{"N"};
    Input<std::int32_t> O{"O"};

    Var k{"k"};
    Var i{"i"}, j{"j"}, o{"o"};
    Var io{"io"}, jo{"jo"}, oo{"oo"};

    DO::Shakti::HalideBackend::SIFT sift;

    Func h;

    Output<Buffer<float>> descriptors{"SIFT", 4};

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
      // TODO: normalize.
      // h(o, j, i, k) = sift.compute_bin_value(i, j, o,             //
      //                                        mag_fn_ext,          //
      //                                        ori_fn_ext,          //
      //                                        x, y, s, scale_max,  //
      //                                        theta);
      // sift.normalize(h, i, j, o, k);
      // descriptors(o, j, i, k) = h(o, j, i, k);
      descriptors(o, j, i, k) = sift.compute_bin_value_v2(  //
          i, j, o,                                          //
          mag_fn_ext,                                       //
          ori_fn_ext,                                       //
          x, y, s, scale_max,                               //
          theta);
    }

    void schedule()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        descriptors.gpu_tile(o, j, i, oo, jo, io, tile_o, tile_j, tile_i,
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

      // // CPU schedule.
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


HALIDE_REGISTER_GENERATOR(SIFT, shakti_sift_descriptor)
