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

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>

#include <DO/Shakti/Halide/BinaryOperators.hpp>
#include <DO/Shakti/Halide/GaussianConvolution.hpp>
#include <DO/Shakti/Halide/Resize.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>


// #define DEBUG


namespace DO { namespace Shakti { namespace HalideBackend {

  //! Computes a pyramid of Gaussians.
  inline auto gaussian_pyramid(
      Sara::ImageView<float>& image,
      const Sara::ImagePyramidParams& params = Sara::ImagePyramidParams(),
      int gaussian_truncation_factor = 4)
      -> Sara::ImagePyramid<float>
  {
    // Resize the image with the appropriate factor.
    //
    // If the octave index is -1, we enlarge the image by a factor 2.
    const auto resize_factor = std::pow(2.f, -params.first_octave_index());
    const auto new_sizes =
        (resize_factor * image.sizes().cast<float>()).cast<int>();

    auto image_start = Sara::Image<float>{new_sizes};

#ifdef DEBUG
    Sara::tic();
#endif
    Shakti::HalideBackend::enlarge(image, image_start);
#ifdef DEBUG
    Sara::toc("Enlarge");
#endif

    // Adjust the new camera sigma with the appropriate factor.
    //
    // Note that the enlarged photograph cannot have a scale of resolution lower
    // than the original image's scale.
    const auto scale_camera = params.scale_camera() * resize_factor;

    // Blur the image so that its scale is equal to the beginning scale of
    // the image pyramid.
    const auto scale_initial = params.scale_initial();
    if (scale_camera < scale_initial)
    {
#ifdef DEBUG
      Sara::tic();
#endif
      const auto sigma = std::sqrt(scale_initial * scale_initial -
                                   scale_camera * scale_camera);
      auto image_start_blurred = image_start;
      Shakti::HalideBackend::gaussian_convolution(
          image_start, image_start_blurred, sigma, gaussian_truncation_factor);
      image_start.swap(image_start_blurred);
#ifdef DEBUG
      Sara::toc("Blur to initial scale of the pyramid");
#endif
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
    const auto num_scales = params.scale_count_per_octave();
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
#ifdef DEBUG
      Sara::tic();
#endif
      auto sigma_s_1 = scale_initial;
      if (o == 0)
        G(0, o) = image_start;
      else
      {
        G(0, o).resize(G(downscale_index, o - 1).sizes() / 2);
        Shakti::HalideBackend::scale(G(downscale_index, o - 1), G(0, o));
      }
#ifdef DEBUG
      Sara::toc("Downscale");
      std::cout << "Current image sizes = " << G(0, o).sizes().transpose()
                << std::endl;
#endif

      for (auto s = 1; s < num_scales; ++s)
      {
        G(s, o).resize(G(0, o).sizes());
        Sara::tic();
        const auto sigma = sigma_s_1 * sqrt(k * k - 1);
        Shakti::HalideBackend::gaussian_convolution(G(s - 1, o), G(s, o), sigma,
                                                    gaussian_truncation_factor);
        sigma_s_1 *= k;
#ifdef DEBUG
        Sara::toc(Sara::format("Convolve (s=%d, o=%d) and kernel sizes = %d", s,
                               o, int(2 * sigma * gaussian_truncation_factor + 1)));
#endif
      }
    }

    return G;
  }

  //! Computes a pyramid of Gaussians.
  inline auto subtract_pyramid(Sara::ImagePyramid<float>& gaussian_pyramid)
      -> Sara::ImagePyramid<float>
  {
    auto& G = gaussian_pyramid;

    auto D = Sara::ImagePyramid<float>{};
    D.reset(G.octave_count(),                //
            G.scale_count_per_octave() - 1,  //
            G.scale_initial(),              //
            G.scale_geometric_factor());    //

    for (auto o = 0; o < D.octave_count(); ++o)
    {
      D.octave_scaling_factor(o) = G.octave_scaling_factor(o);
      for (auto s = 0; s < D.scale_count_per_octave(); ++s)
      {
        D(s, o).resize(G(s, o).sizes());
#ifdef DEBUG
        Sara::tic();
#endif
        Shakti::HalideBackend::subtract(G(s + 1, o), G(s, o), D(s, o));
#ifdef DEBUG
        Sara::toc(Sara::format("Subtracting at (s=%d, o=%d)", s, o));
#endif
      }
    }

    return D;
  }

  //! Computes a pyramid of Gaussians.
  inline auto difference_of_gaussians_pyramid(
      Sara::ImageView<float>& image,
      const Sara::ImagePyramidParams& params = Sara::ImagePyramidParams())
      -> Sara::ImagePyramid<float>
  {
    auto G = Shakti::HalideBackend::gaussian_pyramid(image, params);
    auto D = Shakti::HalideBackend::subtract_pyramid(G);
    return D;
  }

}}}  // namespace DO::Shakti::HalideBackend
