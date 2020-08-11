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

#include <drafts/Halide/Components/Differential.hpp>
#include <drafts/Halide/Components/GaussianKernel.hpp>
#include <drafts/Halide/Components/SeparableConvolution.hpp>
#include <drafts/Halide/Components/DoGExtremum.hpp>


namespace {

  using namespace Halide;
  namespace halide = DO::Shakti::HalideBackend;

  class SIFTOctave : public Halide::Generator<SIFTOctave>
  {
  public:
    GeneratorParam<std::int32_t> tile_x{"tile_x", 32};
    GeneratorParam<std::int32_t> tile_y{"tile_y", 32};

    Input<Buffer<float>> input{"input", 2};
    Input<float> edge_ratio{"edge_ratio"};
    Input<float> extremum_threshold{"extremum_threshold"};

    // Intermediate functions.
    std::vector<halide::GaussianKernel> k;
    std::vector<halide::SeparableConvolution> separable_conv_2d;
    std::vector<Func> g;

    // Output<Buffer<float>[]> gradient_map { "gradients", 2 };
    Output<Buffer<float>[]> dog { "dog", 2 };
    Output<Buffer<std::int8_t>[]> dog_extrema_map { "dog_extrema", 2 };

    Var x{"x"}, y{"y"};
    Var xo{"xo"}, yo{"yo"};
    Var xi{"xi"}, yi{"yi"};

    static constexpr auto num_scales = 3;
    static constexpr auto scale_initial = 1.6f;
    static constexpr auto truncation_factor = 4;

    void generate()
    {
      // Generate scale values.
      const auto scale_exponent = 1.f / num_scales;
      const auto scale_factor = std::pow(2, scale_exponent);
      auto scales = std::vector<float>(num_scales + 3);
      for (auto s = 0; s < 2; ++s)
        scales[s] = scale_initial * std::pow(scale_factor, s);

      // Generate Gaussian standard deviation values.
      auto sigmas = std::vector<float>(num_scales);
      for (auto s = 0; s < num_scales + 2; ++s)
        sigmas[s] = std::sqrt(std::pow(scales[s + 1], 2) - std::pow(scales[s], 2));

      k.resize(num_scales + 2);
      for (auto s = 0; s < num_scales + 2; ++s)
      {
        k[s].kernel = Func{"kernel_" + std::to_string(s)};
        k[s].generate(sigmas[s], truncation_factor);
        k[s].schedule();
      }

      const auto w = input.dim(0).extent();
      const auto h = input.dim(1).extent();

      // Construct the Gaussian octave.
      g.resize(num_scales + 3);

      // Generate the first Gaussian image.
      g[0] = BoundaryConditions::repeat_edge(input);
      g[0].compute_root();

      // Generate the next Gaussian images.
      separable_conv_2d.resize(num_scales + 2);
      for (auto s = 0; s < num_scales + 2; ++s)
      {
        separable_conv_2d[s].generate_2d(                      //
            g[s],                                              //
            k[s].kernel, k[s].kernel_size, k[s].kernel_shift,  //
            k[s].kernel, k[s].kernel_size, k[s].kernel_shift,  //
            g[s + 1], w, h);
        separable_conv_2d[s].schedule_2d(get_target(),    //
                                         tile_x, tile_y,  //
                                         g[s + 1]);
        g[s + 1].compute_root();
      }

      dog.resize(num_scales + 2);
      for (auto s = 0; s < num_scales + 2; ++s)
      {
        dog[s](x, y) = g[s + 1](x, y) - g[s](x, y);
        // dog[s].gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
        //                 TailStrategy::GuardWithIf);
      }

      dog_extrema_map.resize(num_scales);
      for (auto s = 1; s < num_scales + 1;++s)
      {
        dog_extrema_map[s - 1](x, y) = halide::is_dog_extremum(  //
            g[s - 1], g[s], g[s + 1],                            //
            edge_ratio, extremum_threshold,                      //
            x, y);
        // dog_extrema_map[s - 1].gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
        //                                 TailStrategy::GuardWithIf);
      }

      // gradients.resize(num_scales);
      // for (auto s = 0; s < num_scales; ++s)
      // {
      //   gradients[s] = halide::gradient(g[s + 1], x, y);
      //   gradients[s].gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
      //                         TailStrategy::GuardWithIf);
      // }
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(SIFTOctave, shakti_sift_octave)
