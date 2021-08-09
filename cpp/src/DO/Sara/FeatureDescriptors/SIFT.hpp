// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Sara/Geometry/Tools/Utilities.hpp>

#include <DO/Sara/Graphics.hpp>

#include <DO/Sara/Features/Feature.hpp>

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif


namespace DO { namespace Sara {

  /*!
   *  @addtogroup FeatureDescriptors
   *  @{
   */

  //! @brief Functor class used to compute the SIFT Descriptor at some location.
  template <int N = 4, int O = 8>
  class ComputeSIFTDescriptor
  {
  public: /* interface. */
    static constexpr auto Dim = N * N * O;

    using descriptor_type = Matrix<float, Dim, 1>;

    //! @brief Constructor.
    inline ComputeSIFTDescriptor(float bin_scale_unit_length = 3.f,
                                 float max_bin_value = 0.2f)
      : _bin_scale_unit_length{bin_scale_unit_length}
      , _max_bin_value{max_bin_value}
    {
    }

    //! @brief Computes the SIFT descriptor for keypoint @f$ (x,y,\sigma,\theta) @f$.
    auto operator()(float x, float y, float sigma, float theta,
                    const ImageView<Vector2f>& grad_polar_coords,
                    bool do_normalization = true) const
        -> descriptor_type
    {
      constexpr auto pi = static_cast<float>(M_PI);

      // The radius of each overlapping patches.
      const auto& lambda = _bin_scale_unit_length;
      const auto l = lambda * sigma;

      // The radius of the total patch.
      const auto r = sqrt(2.f) * l * (N + 1) / 2.f;

      // Linear part of the patch normalization transform.
      auto T = Matrix2f{};
      T << cos(theta), sin(theta),
          -sin(theta), cos(theta);
      T /= l;

      // The SIFT descriptor.
      descriptor_type h = descriptor_type::Zero();

      // Loop to perform interpolation
      const int rounded_r = int_round(r);
      const int rounded_x = int_round(x);
      const int rounded_y = int_round(y);

      for (auto v = -rounded_r; v <= rounded_r; ++v)
      {
        for (auto u = -rounded_r; u <= rounded_r; ++u)
        {
          // Retrieve the coordinates in the normalized patch coordinate frame.
          auto pos = Vector2f{T * Vector2f(u, v)};

          // Boundary check.
          if (rounded_x + u < 0 || rounded_x + u >= grad_polar_coords.width() ||
              rounded_y + v < 0 || rounded_y + v >= grad_polar_coords.height())
            continue;

          // Compute the Gaussian weight which gives less emphasis to gradients
          // further from the keypoint center.
          constexpr auto sigma = N * N * 0.25f;  // (N / 2)^2
          const auto weight = std::exp(-pos.squaredNorm() / (2.f * sigma));

          // Read the precomputed gradient (in polar coordinates).
          const auto& mag = grad_polar_coords(rounded_x + u, rounded_y + v)(0);
          // Notice here the reoriented gradient orientation w.r.t. the dominant
          // gradient orientation.
          auto ori = grad_polar_coords(rounded_x + u, rounded_y + v)(1) - theta;

          // Rescale the orientation to the interval [0, O[.
          ori = ori < 0.f ? ori + 2.f * pi : ori;
          ori *= static_cast<float>(O) / (2.f * pi);

          // Shift the coordinates to retrieve the "SIFT" coordinate system so
          // that $(x,y)$ is in $[-1, N]^2$.
          pos.array() += N / 2.f - 0.5f;

          // Discard pixels that are not in the oriented patch.
          if (pos.minCoeff() <= -1.f || pos.maxCoeff() >= static_cast<float>(N))
            continue;

          // Accumulate the histogram bins using trilinear interpolation.
          accumulate(h, pos, ori, weight, mag);
        }
      }

      if (do_normalization)
      {
        normalize(h);
        h = (h * 512.f).cwiseMin(Matrix<float, Dim, 1>::Ones() * 255.f);
      }

      return h;

    }

    //! @brief Computes the **upright** SIFT descriptor for keypoint
    //! @f$(x,y,\sigma) @f$.
    auto operator()(float x, float y, float sigma,
                    const ImageView<Vector2f>& grad_polar_coords) const
        -> descriptor_type
    {
      return this->operator()(x, y, sigma, 0.f, grad_polar_coords);
    }

    //! Helper member function.
    auto operator()(const OERegion& f,
                    const ImageView<Vector2f>& grad_polar_coords) const
        -> descriptor_type
    {
      return this->operator()(f.x(), f.y(), f.scale(), f.orientation,
                              grad_polar_coords);
    }

    //! Helper member function.
    auto operator()(const std::vector<OERegion>& features,
                    const std::vector<Point2i>& scale_octave_pairs,
                    const ImagePyramid<Vector2f>& gradient_polar_coords,
                    bool parallel = false) const
        -> Tensor_<float, 2>
    {
      auto sifts = Tensor_<float, 2>{{int(features.size()), Dim}};
      if (parallel)
      {
        const auto num_features = static_cast<int>(features.size());
#pragma omp parallel for
        for (auto i = 0; i < num_features; ++i)
        {
          sifts.matrix().row(i) =
              this->operator()(features[i],
                               gradient_polar_coords(scale_octave_pairs[i](0),
                                                     scale_octave_pairs[i](1)))
                  .transpose();
        }
      }
      else
      {
        for (size_t i = 0; i < features.size(); ++i)
        {
          sifts.matrix().row(i) =
              this->operator()(features[i],
                               gradient_polar_coords(scale_octave_pairs[i](0),
                                                     scale_octave_pairs[i](1)))
                  .transpose();
        }
      }
      return sifts;
    }

  private: /* member functions. */
    //! The accumulation function based on trilinear interpolation.
    void accumulate(descriptor_type& h, const Vector2f& pos, float ori,
                    float weight, float mag) const
    {
      float xif, yif, oriif;
      const auto xfrac = std::modf(pos.x(), &xif);
      const auto yfrac = std::modf(pos.y(), &yif);
      const auto orifrac = std::modf(ori, &oriif);
      const auto xi = int(xif);
      const auto yi = int(yif);
      const auto orii = int(oriif);

      for (auto dy = 0; dy < 2; ++dy)
      {
        const auto y = yi + dy;
        if (y < 0 || y >= N)
          continue;

        const auto wy = (dy == 0) ? 1 - yfrac : yfrac;
        for (auto dx = 0; dx < 2; ++dx)
        {
          const auto x = xi + dx;
          if (x < 0 || x >= N)
            continue;

          const auto wx = (dx == 0) ? 1 - xfrac : xfrac;
          for (auto dori = 0; dori < 2; ++dori)
          {
            const auto o = (orii + dori) % O;
            const auto wo = (dori == 0) ? 1 - orifrac : orifrac;

            h[at(y, x, o)] += wy * wx * wo * weight * mag;
          }
        }
      }
    }

    //! @brief Normalize in a contrast-invariant way.
    void normalize(descriptor_type& h) const
    {
      // Euclidean normalization to account for contrast changes.
      h.normalize();

      // Make the descriptor robustness to nonlinear illumination changes.
      //
      // 1) Clamp histogram bin values to 0.2. (as indicated in the paper).
      h = h.cwiseMin(descriptor_type::Ones() * _max_bin_value);
      // 2) Renormalize again.
      h.normalize();
    }

    //! Helper access function.
    inline int at(int i, int j, int o) const
    {
      return N * O * i + j * O + o;
    }

  private: /* data members. */
    float _bin_scale_unit_length;
    float _max_bin_value;
  };

  //! @}


} /* namespace Sara */
} /* namespace DO */
