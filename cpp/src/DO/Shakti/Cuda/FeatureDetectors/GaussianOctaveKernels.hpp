// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Math/UsualFunctions.hpp>
#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Shakti::Cuda {

  template <typename T>
  struct GaussianOctaveKernels
  {
    inline GaussianOctaveKernels(int scale_count = 3, T scale_camera = T{1},
                                 T scale_initial = T(1.6),
                                 T gaussian_truncation_factor = T(4))
      : scale_count{scale_count}
      , scale_camera{scale_camera}
      , scale_initial{scale_initial}
      , gaussian_truncation_factor{gaussian_truncation_factor}
      , scale_factor{std::pow(2.f, 1.f / scale_count)}
    {
      // List the scales in the discrete octave.
      scales = std::vector<T>(scale_count + 3);
      for (auto i = 0; i < scale_count + 3; ++i)
        scales[i] = scale_initial * std::pow(scale_factor, i);

      // Calculate the Gaussian smoothing values.
      sigmas = std::vector<T>(scale_count + 3);
      for (auto i = 0u; i < sigmas.size(); ++i)
      {
        sigmas[i] = i == 0 ? std::sqrt(Sara::square(scales[0]) -
                                       Sara::square(scale_camera))
                           : std::sqrt(Sara::square(scales[i]) -
                                       Sara::square(scales[i - 1]));
      }

      SARA_DEBUG << "scales =\n"
                 << Eigen::Map<const Eigen::RowVectorXf>(scales.data(),
                                                         scales.size())
                 << std::endl;
      SARA_DEBUG << "sigmas =\n"
                 << Eigen::Map<const Eigen::RowVectorXf>(sigmas.data(),
                                                         sigmas.size())
                 << std::endl;

      // Calculate the kernel dimensions.
      kernel_radii = std::vector<int>{};
      for (const auto& sigma : sigmas)
      {
        auto radius = std::round(gaussian_truncation_factor * sigma);
        if (radius == 0)
          radius = 1;
        kernel_radii.push_back(static_cast<int>(radius));
      }
      const auto kernel_radius_max = kernel_radii.back();
      SARA_DEBUG << "kernel radii =\n"
                 << Eigen::Map<const Eigen::RowVectorXi>(kernel_radii.data(),
                                                         kernel_radii.size())
                 << std::endl;

      // Fill the Gaussian kernels by storing the positive part only since the
      // Gaussian kernel is symmetric.
      kernels = Sara::Tensor_<T, 2>{scale_count + 3, kernel_radius_max};
      kernels.flat_array().fill(0);

      for (auto n = 0; n < kernels.size(0); ++n)
      {
        const auto& sigma = sigmas[n];
        const auto two_sigma_squared = 2 * Sara::square(sigma);
        const auto kernel_radius = kernel_radii[n];

        for (auto k = 0; k <= kernel_radius; ++k)
          kernels(n, k) = exp(-Sara::square(k) / two_sigma_squared);

        auto kernel_sum = kernels(n, 0);
        for (auto k = 1; k <= kernel_radius; ++k)
          kernel_sum += 2 * kernels(n, k);

        for (auto k = 0; k <= kernel_radius; ++k)
          kernels(n, k) /= kernel_sum;
      }

      Eigen::IOFormat HeavyFmt(3, 0, ", ", ",\n", "[", "]", "[", "]");
      SARA_CHECK(Eigen::Map<const Eigen::RowVectorXf>(  //
          sigmas.data(),                                //
          sigmas.size())                                //
      );
      SARA_CHECK(kernels.sizes().reverse().transpose());
      SARA_DEBUG << "stacked kernels =\n"
                 << kernels.matrix().transpose().format(HeavyFmt) << std::endl;
    }

    int scale_count = 3;
    float scale_camera = 1.f;
    float scale_initial = 1.6f;
    float gaussian_truncation_factor = 4.f;
    float scale_factor = std::pow(2.f, 1.f / 3);

    std::vector<T> scales;
    std::vector<T> sigmas;
    std::vector<int> kernel_radii;
    Sara::Tensor_<T, 2> kernels;
  };

}  // namespace DO::Shakti::Cuda
