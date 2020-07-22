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

  class SIFT
    : public Generator<SIFT>
  {
  public:
    GeneratorParam<int> tile_o{"tile_x", 16};
    GeneratorParam<int> tile_k{"tile_y", 16};

    // Input data.
    Input<Buffer<float>[2]> polar_gradient{"gradients", 2};
    Input<Buffer<float>[4]> xyst{"xyst", 4};
    Input<float> scale_max{"scale_max"};
    Input<float> bin_length_in_scale_unit{"bin_length_in_scale_unit"};
    Input<std::int32_t> N{"N"};
    Input<std::int32_t> O{"O"};


    Var i{"i"}, j{"j"}, o{"o"}, k{"k"};
    Var io{"i"}, jo{"j"}, oo{"oo"}, ko{"ko"};
    Var ii{"i"}, ji{"j"}, oi{"oi"}, ki{"ki"};

    DO::Shakti::HalideBackend::SIFT sift;

    Func h;
    Func h_final;

    Output<Buffer<float>> descriptors;

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
      h(i, j, o, k) = sift.compute_bin_value(i, j, o, mag_fn_ext, ori_fn_ext,
                                             x, y, s, scale_max, theta);

      sift.normalize(h, i, j, o, k);

      descriptors(i, j, o, k) = h_final(i, j, o, k);
    }

    void schedule()
    {
      using namespace DO::Shakti::HalideBackend;

    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(SIFT, shakti_sift_descriptor)
