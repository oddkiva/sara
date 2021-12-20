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

#include <DO/Sara/Core/Math/UsualFunctions.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Sara/ImageProcessing/Differential.hpp>
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup ScaleSpace
    @{
   */

  //! Computes a pyramid of Gaussians.
  template <typename T>
  inline auto
  gaussian_pyramid(const ImageView<T>& image,
                   const ImagePyramidParams& params = ImagePyramidParams())
      -> ImagePyramid<T>
  {
    using Scalar = typename ImagePyramid<T>::scalar_type;

    // Resize the image with the appropriate factor.
    const auto resize_factor =
        std::pow(2.f, -static_cast<float>(params.first_octave_index()));

    // Deduce the new camera sigma with respect to the dilated image.
    const auto camera_sigma = Scalar(params.scale_camera()) * resize_factor;

    // Blur the image so that its new sigma is equal to the initial sigma.
    const auto init_sigma = Scalar(params.scale_initial());

    auto I = Image<T>{};
    if (params.first_octave_index() < 0)
      I = enlarge(image, resize_factor);
    else if (params.first_octave_index() > 0)
    {
      if (camera_sigma < init_sigma)
      {
        const auto sigma = std::sqrt(square(init_sigma) - square(camera_sigma));
        I = gaussian(image, sigma);
      }
      else
        I = image;
      I = downscale(I, int(std::round(1 / resize_factor)));
    }
    else
    {
      if (camera_sigma < init_sigma)
      {
        const auto sigma = std::sqrt(square(init_sigma) - square(camera_sigma));
        I = gaussian(image, sigma);
      }
      else
        I = image;
    }

    // Deduce the maximum number of octaves.
    // 1. Determine the min of the image image sizes `l`.
    const auto l = std::min(I.width(), I.height());
    // 2. The minimum image size in each dimension `b`.
    const auto b = params.image_padding_size();
    // 3. The number of octaves is determined as follows.
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
    const auto k = static_cast<Scalar>(params.scale_geometric_factor());
    const auto num_scales = params.num_scales_per_octave();
    const auto downscale_index = static_cast<int>(floor(log(Scalar(2)) / log(k)));

    // Create the image pyramid
    auto G = ImagePyramid<T>{};
    G.reset(num_octaves, num_scales, init_sigma, k);

    for (auto o = 0; o < num_octaves; ++o)
    {
      // Compute the octave scaling factor
      G.octave_scaling_factor(o) =
          (o == 0) ? 1 / resize_factor : G.octave_scaling_factor(o - 1) * 2;

      // Compute the gaussians in octave @f$o@f$
      auto sigma_s_1 = init_sigma;
      G(0, o) = o == 0 ? std::move(I) : downscale(G(downscale_index, o - 1), 2);

      for (auto s = 1; s < num_scales; ++s)
      {
        const auto sigma = sqrt(square(k * sigma_s_1) - square(sigma_s_1));
        G(s, o) = gaussian(G(s - 1, o), sigma);
        sigma_s_1 *= k;
      }
    }

    return G;
  }

  auto difference_of_gaussians_pyramid(const ImagePyramid<float>& gaussians)
      -> ImagePyramid<float>;

  //! Computes a pyramid of difference of Gaussians from the Gaussian pyramid.
  template <typename T>
  inline auto difference_of_gaussians_pyramid(const ImagePyramid<T>& gaussians)
      -> ImagePyramid<T>
  {
    auto D = ImagePyramid<T>{};
    D.reset(gaussians.num_octaves(), gaussians.num_scales_per_octave() - 1,
            gaussians.scale_initial(), gaussians.scale_geometric_factor());

    for (auto o = 0; o < D.num_octaves(); ++o)
    {
      D.octave_scaling_factor(o) = gaussians.octave_scaling_factor(o);
      for (auto s = 0; s < D.num_scales_per_octave(); ++s)
      {
        D(s, o).resize(gaussians(s, o).sizes());
        tensor_view(D(s, o)).flat_array() =
            tensor_view(gaussians(s + 1, o)).flat_array() -
            tensor_view(gaussians(s, o)).flat_array();
      }
    }
    return D;
  }

  //! Computes a pyramid of scale-normalized Laplacians of Gaussians from the
  //! Gaussian pyramid.
  template <typename T>
  inline auto laplacian_pyramid(const ImagePyramid<T>& gaussians)
      -> ImagePyramid<T>
  {
    using scalar_type = typename ImagePyramid<T>::scalar_type;

    auto LoG = ImagePyramid<T>{};
    LoG.reset(gaussians.num_octaves(), gaussians.num_scales_per_octave(),
              gaussians.scale_initial(), gaussians.scale_geometric_factor());

    for (auto o = 0; o < LoG.num_octaves(); ++o)
    {
      LoG.octave_scaling_factor(o) = gaussians.octave_scaling_factor(o);
      for (auto s = 0; s < LoG.num_scales_per_octave(); ++s)
      {
        LoG(s, o) = laplacian(gaussians(s, o));
        for (auto& p : LoG(s, o))
          p *= static_cast<scalar_type>(
              Sara::square(gaussians.scale_relative_to_octave(s)));
      }
    }

    return LoG;
  }

  //! Computes the gradient vector of @f$I(x,y,\sigma)@f$ at @f$(x,y,\sigma)@f$,
  //! where @f$\sigma = 2^{s/S + o}@f$ where @f$S@f$ is the number of scales per
  //! octave.
  template <typename T>
  inline auto gradient(const ImagePyramid<T>& I, int x, int y, int s, int o)
      -> Matrix<T, 3, 1>
  {
    if (x < 1 || x >= I(s, o).width() - 1 || y < 1 ||
        y >= I(s, o).height() - 1 || s < 1 ||
        s >= static_cast<int>(I(o).size()) - 1)
      throw std::out_of_range{"Computing gradient out of image range!"};

    auto d = Matrix<T, 3, 1>{};
    d(0) = (I(x + 1, y, s, o) - I(x - 1, y, s, o)) / T(2);
    d(1) = (I(x, y + 1, s, o) - I(x, y - 1, s, o)) / T(2);
    d(2) = (I(x, y, s + 1, o) - I(x, y, s - 1, o)) / T(2);
    return d;
  }

  //! Computes the hessian matrix of @f$I(x,y,\sigma)@f$ at @f$(x,y,\sigma)@f$,
  //! where @f$\sigma = 2^{s/S + o}@f$ where @f$S@f$ is the number of scales
  //! per octave.
  template <typename T>
  inline auto hessian(const ImagePyramid<T>& I, int x, int y, int s, int o)
      -> Matrix<T, 3, 3>
  {
    if (x < 1 || x >= I(s, o).width() - 1 || y < 1 ||
        y >= I(s, o).height() - 1 || s < 1 ||
        s >= static_cast<int>(I(o).size()) - 1)
      throw std::out_of_range{"Computing Hessian matrix out of image range!"};

    auto H = Matrix<T, 3, 3>{};

    // Ixx
    H(0, 0) = I(x + 1, y, s, o) - T(2) * I(x, y, s, o) + I(x - 1, y, s, o);
    // Iyy
    H(1, 1) = I(x, y + 1, s, o) - T(2) * I(x, y, s, o) + I(x, y - 1, s, o);
    // Iss
    H(2, 2) = I(x, y, s + 1, o) - T(2) * I(x, y, s, o) + I(x, y, s - 1, o);
    // Ixy = Iyx
    H(0, 1) = H(1, 0) = (I(x + 1, y + 1, s, o) - I(x - 1, y + 1, s, o) -
                         I(x + 1, y - 1, s, o) + I(x - 1, y - 1, s, o)) /
                        T(4);
    // Ixs = Isx
    H(0, 2) = H(2, 0) = (I(x + 1, y, s + 1, o) - I(x - 1, y, s + 1, o) -
                         I(x + 1, y, s - 1, o) + I(x - 1, y, s - 1, o)) /
                        T(4);
    // Iys = Isy
    H(1, 2) = H(2, 1) = (I(x, y + 1, s + 1, o) - I(x, y - 1, s + 1, o) -
                         I(x, y + 1, s - 1, o) + I(x, y - 1, s - 1, o)) /
                        T(4);
    // Done!
    return H;
  }

  //! @}

}}  // namespace DO::Sara
