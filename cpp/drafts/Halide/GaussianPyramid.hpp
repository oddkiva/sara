// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Sara/ImageProcessing/Differential.hpp>
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <drafts/Halide/BinaryOperators.hpp>
#include <drafts/Halide/GaussianConvolution.hpp>
#include <drafts/Halide/Resize.hpp>
#include <drafts/Halide/Utilities.hpp>


namespace DO { namespace Shakti { namespace HalideBackend {

  //! Computes a pyramid of Gaussians.
  inline auto gaussian_pyramid(
      Sara::ImageView<float>& image,
      const Sara::ImagePyramidParams& params = Sara::ImagePyramidParams())
      -> Sara::ImagePyramid<float>
  {
    // Resize the image with the appropriate factor.
    //
    // If the octave index is -1, we enlarge the image by a factor 2.
    const auto resize_factor = std::pow(2.f, -params.first_octave_index());
    const auto new_sizes =
        (resize_factor * image.sizes().cast<float>()).cast<int>();

    auto image_start = Sara::Image<float>{new_sizes};

    Sara::tic();
    Shakti::HalideBackend::enlarge(image, image_start);
    Sara::toc("Enlarge");

    // Adjust the new camera sigma with the appropriate factor.
    //
    // Note that the enlarged photograph cannot have a lower scale of resolution
    // than the original image.
    const auto scale_camera = params.scale_camera() * resize_factor;

    // Blur the image so that its scale is equal to the beginning scale of
    // the image pyramid.
    const auto scale_initial = params.scale_initial();
    if (scale_camera < scale_initial)
    {
      Sara::tic();
      const auto sigma = std::sqrt(scale_initial * scale_initial -
                                   scale_camera * scale_camera);
      auto image_start_blurred = image_start;
      gaussian_convolution(image_start, image_start_blurred, sigma, 4);
      image_start.swap(image_start_blurred);
      Sara::toc("Blur to initial scale of the pyramid");
    }

    // Deduce the maximum number of octaves.
    const auto l = std::min(image.width(),
                            image.height());     // l = min image image sizes.
    const auto b = params.image_padding_size();  // b = image border size.

    /*
     * Calculation details:
     *
     * We must satisfy:
     *   l / 2^k > 2b
     *   2^k < l / (2b)
     *   k < log(l / (2b)) / log(2)
     *
     */
    const auto num_octaves = static_cast<int>(log(l / (2.f * b)) / log(2.f));

    // Shorten names.
    const auto k = params.scale_geometric_factor();
    const auto num_scales = params.num_scales_per_octave();
    const auto downscale_index = static_cast<int>(floor(log(2) / log(k)));

    // Create the image pyramid
    auto G = Sara::ImagePyramid<float>{};
    G.reset(num_octaves, num_scales, scale_initial, k);

    for (auto o = 0; o < num_octaves; ++o)
    {
      // Compute the octave scaling factor
      G.octave_scaling_factor(o) =
          (o == 0) ? 1.f / resize_factor : G.octave_scaling_factor(o - 1) * 2;

      // Compute the gaussians in octave @f$o@f$
      Sara::tic();
      auto sigma_s_1 = scale_initial;
      if (o == 0)
        G(0, o) = image_start;
      else
      {
        G(0, o).resize(G(downscale_index, o - 1).sizes() / 2);
        Shakti::HalideBackend::scale(G(downscale_index, o - 1), G(0, o));
      }
      Sara::toc("Downscale");
      std::cout << "Current image sizes = " << G(0, o).sizes().transpose()
                << std::endl;

      for (auto s = 1; s < num_scales; ++s)
      {
        G(s, o).resize(G(0, o).sizes());
        Sara::tic();
        const auto sigma =
            sqrt(k * k * sigma_s_1 * sigma_s_1 - sigma_s_1 * sigma_s_1);
        gaussian_convolution(G(s - 1, o), G(s, o), sigma, 4);
        sigma_s_1 *= k;
        Sara::toc(Sara::format("Convolve (s=%d, o=%d) and kernel sizes = %d", s,
                               o, int(2 * sigma * 4 + 1)));
      }
    }

    return G;
  }

  //! Computes a pyramid of Gaussians.
  inline auto difference_of_gaussians_pyramid(
      Sara::ImageView<float>& image,
      const Sara::ImagePyramidParams& params = Sara::ImagePyramidParams())
      -> Sara::ImagePyramid<float>
  {
    auto G = gaussian_pyramid(image, params);

    auto D = Sara::ImagePyramid<float>{};
    D.reset(G.num_octaves(),                //
            G.num_scales_per_octave() - 1,  //
            G.scale_initial(),              //
            G.scale_geometric_factor());    //

    for (auto o = 0; o < D.num_octaves(); ++o)
    {
      D.octave_scaling_factor(o) = G.octave_scaling_factor(o);
      for (auto s = 0; s < D.num_scales_per_octave(); ++s)
      {
        D(s, o).resize(G(s, o).sizes());
        Sara::tic();
        subtract(G(s + 1, o), G(s, o), D(s, o));
        Sara::toc(Sara::format("Subtracting at (s=%d, o=%d)", s, o));
      }
    }

    return D;
  }

}}}  // namespace DO::Shakti::HalideBackend
