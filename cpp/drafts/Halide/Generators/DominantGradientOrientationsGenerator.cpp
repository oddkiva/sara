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

#include <drafts/Halide/Components/DominantGradientOrientations.hpp>


namespace {

  using namespace Halide;

  class DominantGradientOrientations
    : public Generator<DominantGradientOrientations>
  {
  public:
    GeneratorParam<int> tile_o{"tile_x", 16};
    GeneratorParam<int> tile_k{"tile_y", 16};

    // Input data.
    Input<Buffer<float>[2]> polar_gradient{"gradients", 2};
    Input<Buffer<float>[3]> xys{"xys", 1};
    Input<float> scale_upper_bound{"scale_initial"};

    // Algorithm parameters.
    Input<std::int32_t> num_orientation_bins{"num_orientation_bins"};
    Input<float> gaussian_truncation_factor{"gaussian_truncation_factor"};
    Input<float> scale_mult_factor{"scale_multiplying_factor"};
    Input<float> peak_ratio_thres{"peak_ratio_thres"};

    // HoG.
    Func hog;

    // Iterated box-blurred HoG.
    static constexpr auto num_blur_iterations = 6;
    std::array<Func, num_blur_iterations> hog_blurred;

    // Peak maps.
    Output<Buffer<bool>> peak_map{"peak_map", 2};
    // Peak residuals.
    Output<Buffer<float>> peak_residuals{"peak_residuals", 2};

    Var o{"o"}, k{"k"};
    Var oo{"oo"}, ko{"ko"};
    Var oi{"oi"}, ki{"ki"};

    void generate()
    {
      const auto& mag = polar_gradient[0];
      const auto& ori = polar_gradient[1];
      const auto& mag_fn_ext = BoundaryConditions::constant_exterior(mag, 0);
      const auto& ori_fn_ext = BoundaryConditions::constant_exterior(ori, 0);

      const auto x = xys[0](k);
      const auto y = xys[1](k);
      const auto s = xys[2](k);

      namespace halide = DO::Shakti::HalideBackend;
      hog(o, k) = halide::compute_histogram_of_gradients(  //
          mag_fn_ext, ori_fn_ext,                          //
          x, y, s,                                         //
          scale_upper_bound,                               //
          o,                                               //
          num_orientation_bins,                            //
          gaussian_truncation_factor,                      //
          scale_mult_factor);
      hog.compute_root();

      for (auto i = 0; i < num_blur_iterations; ++i)
      {
        hog_blurred[i] = Func{"hog_blurred" + std::to_string(i)};
        auto& prev = i == 0 ? hog : hog_blurred[i - 1];
        hog_blurred[i](o, k) = halide::box_blur(prev,                   //
                                                o, k,                   //
                                                num_orientation_bins);  //
      }
      for (auto i = 0; i < num_blur_iterations; ++i)
        hog_blurred[i].compute_root();

      const auto& hog_final = hog_blurred.back();

      peak_map(o, k) = halide::is_peak(hog_final,             //
                                       o, k,                  //
                                       num_orientation_bins,  //
                                       peak_ratio_thres);
      peak_map.compute_root();

      peak_residuals(o, k) = halide::compute_peak_residual_map(  //
          hog_final,                                             //
          peak_map,                                              //
          o, k,                                                  //
          num_orientation_bins);
    }

    void schedule()
    {
      using namespace DO::Shakti::HalideBackend;

      schedule_histograms(hog, o, k,       //
                          oo, ko,          //
                          oi, ki,          //
                          tile_o, tile_k,  //
                          get_target());

      for (auto i = 0; i < num_blur_iterations; ++i)
        schedule_histograms(hog_blurred[i],  //
                            o, k,            //
                            oo, ko,          //
                            oi, ki,          //
                            tile_o, tile_k,  //
                            get_target());

      schedule_histograms(peak_map,        //
                          o, k,            //
                          oo, ko,          //
                          oi, ki,          //
                          tile_o, tile_k,  //
                          get_target());

      schedule_histograms(peak_residuals,  //
                          o, k,            //
                          oo, ko,          //
                          oi, ki,          //
                          tile_o, tile_k,  //
                          get_target());
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(DominantGradientOrientations,
                          shakti_dominant_gradient_orientations)
