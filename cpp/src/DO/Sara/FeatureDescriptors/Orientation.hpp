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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/Core/StringFormat.hpp>

#include <DO/Sara/Features/Feature.hpp>

#include <DO/Sara/Geometry/Tools/Utilities.hpp>

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>


namespace DO { namespace Sara {

  /*!
   *  @ingroup FeatureDescriptors
   *  @defgroup Orientation Dominant Orientation Assignment
   *  @{
  */

  //! @brief Computes the image gradients of image @f$I@f$ in polar coordinates.
  /*!
      In each pixel @f$(x,y) \in [0,w[ \times [0,h[@f$,
      @f$\nabla I(x,y)@f$ is the 2D vector @f$(r,\theta)@f$ where:
      - @f$r = 2 ||\nabla I(x,y)||@f$,
      - @f$\theta = \mathrm{angle}( \nabla I(x,y) )@f$.
   */
  template <typename T>
  Image<Matrix<T,2,1>> gradient_polar_coordinates(const ImageView<T>& f)
  {
    auto nabla_f = gradient(f);
    for (auto it = nabla_f.begin(); it != nabla_f.end(); ++it)
    {
      const auto r = 2 * it->norm();
      const auto theta = atan2(it->y(), it->x());
      *it << r, theta;
    }
    return nabla_f;
  }

  //! @brief Computes the image gradients in polar coordinates for each image in
  //! the pyramid.
  template <typename T>
  ImagePyramid<Matrix<T, 2, 1>> gradient_polar_coordinates(const ImagePyramid<T>& pyramid)
  {
    auto gradient_pyramid = ImagePyramid<Matrix<T, 2, 1>>{};
    gradient_pyramid.reset(
      pyramid.num_octaves(),
      pyramid.num_scales_per_octave(),
      pyramid.scale_initial(),
      pyramid.scale_geometric_factor() );

    for (int o = 0; o < pyramid.num_octaves(); ++o)
    {
      gradient_pyramid.octave_scaling_factor(o) = pyramid.octave_scaling_factor(o);
      for (int s = 0; s < pyramid.num_scales_per_octave(); ++s)
        gradient_pyramid(s,o) = gradient_polar_coordinates(pyramid(s,o));
    }
    return gradient_pyramid;
  }

  //! @brief Computes the orientation histogram on a local patch around keypoint
  //! @f$(x,y,\sigma)@f$.
  template <typename T, int N>
  void compute_orientation_histogram(Array<T, N, 1>& orientation_histogram,
                                     const ImageView<Matrix<T,2,1>>& grad_polar_coords,
                                     T x, T y, T s,
                                     T patch_truncation_factor = T(3),
                                     T blur_factor = T(1.5))
  {
    // Weighted histogram of gradients.
    orientation_histogram.setZero();

    // Rounding of the coordinates.
    auto rounded_x = int_round(x);
    auto rounded_y = int_round(y);

    // std deviation of the gaussian weight (cf. [Lowe, IJCV 2004])
    auto sigma = s * blur_factor;

    // Patch radius on which the histogram of gradients is performed.
    auto patch_radius = int_round(sigma * patch_truncation_factor);

    // Accumulate the histogram of orientations.
    for (auto v = -patch_radius; v <= patch_radius; ++v)
    {
      for (auto u = -patch_radius; u <= patch_radius; ++u)
      {
        if (rounded_x + u < 0 || rounded_x + u >= grad_polar_coords.width() ||
            rounded_y + v < 0 || rounded_y + v >= grad_polar_coords.height())
          continue;

        auto mag = grad_polar_coords(rounded_x + u, rounded_y + v)(0);
        auto ori = grad_polar_coords(rounded_x + u, rounded_y + v)(1);

        // ori is in \f$]-\pi, \pi]\f$, so translate ori by \f$2*\pi\f$ if it is
        // negative.
        ori = ori < 0 ? ori + T(2.*M_PI) : ori;
        auto bin_index = int(floor(ori / T(2 * M_PI) * N));
        bin_index %= N;

        // Give more emphasis to gradient orientations that lie closer to the
        // keypoint location.
        auto weight = exp(-(u * u + v * v) / (T(2) * sigma * sigma));
        // Also give more emphasis to gradient with large magnitude.
        orientation_histogram(bin_index) += weight * mag;
      }
    }
  }

  //! @brief This is used in [Lowe, IJCV 2004] to determine keypoint
  //! orientations.
  /*!
      Basically, the histogram is convolved 6 times with the mean kernel
      \f$[1/3, 1/3, 1/3]\f$.
   */
  template <typename T, int N>
  void lowe_smooth_histogram(Array<T, N, 1>& orientation_histogram,
                             int num_iters = 6)
  {
    // Smooth in place: it works.
    for (int iter = 0; iter < num_iters; ++iter)
    {
      const auto first = orientation_histogram(0);
      auto prev = orientation_histogram(N - 1);
      for (int i = 0; i < N - 1; ++i)
      {
        const auto val =
            (prev + orientation_histogram(i) + orientation_histogram(i + 1)) /
            3.f;
        prev = orientation_histogram(i);
        orientation_histogram(i) = val;
      }
      orientation_histogram(N - 1) =
          (prev + orientation_histogram(N - 1) + first) / 3.f;
    }
  }

  //! @brief This is used in [Lowe, IJCV 2004] to find histogram peaks.
  /*!
      A histogram peak is by definition the index \f$i\f$ such that:
      - \f$h_{i} > h_{i+1}\f$
      - \f$h_{i} > h_{i-1}\f$

      Only histogram peaks \f$i\f$ such that \f$h_i \geq 0.8 \max_j h_j\f$
   */
  template <typename T, int N>
  std::vector<int> find_peaks(const Array<T, N, 1>& orientation_histogram,
                              T peak_ratio_thres = T(0.8))
  {
    T max = orientation_histogram.maxCoeff();
    std::vector<int> orientation_peaks;
    orientation_peaks.reserve(N);
    for (int i = 0; i < N; ++i)
      if (orientation_histogram(i) >= peak_ratio_thres * max &&
          orientation_histogram(i) > orientation_histogram((i - 1 + N) % N) &&
          orientation_histogram(i) > orientation_histogram((i + 1) % N))
        orientation_peaks.push_back(i);
    return orientation_peaks;
  }

  //! @brief Refine peaks as in [Lowe, IJCV 2004] by interpolation based on a
  //! second-order Taylor approximation.
  template <typename T, int N>
  T refine_peak(const Array<T, N, 1>& orientation_histogram, int i)
  {
    const auto y0 = orientation_histogram((i - 1 + N) % N);
    const auto y1 = orientation_histogram(i);
    const auto y2 = orientation_histogram((i + 1) % N);

    // Denote the orientation histogram function by \f$f\f$.
    // perform a 2nd-order Taylor approximation:
    // \f$f(x+h) = f(x) + f'(x)h + f''(x) h^2/2\f$
    // We approximate \f$f'\f$ and \f$f''\f$ by finite difference.
    const auto fprime = (y2 - y0) / 2.f;
    const auto fsecond = y0 - 2.f * y1 + y2;

    // Maximize w.r.t. to \f$h\f$, derive the expression.
    // Thus \f$h = -f'(x)/f''(x)\f$.
    const auto h = -fprime / fsecond;

    // Add the offset \f$h\f$ to get the refined orientation value.
    // Note that we also add the 0.5f offset, because samples are assumed taken
    // on the middle of the interval \f$[i, i+1)\f$.
    return T(i) + T(0.5) + h;
  }

  //! @brief Helper functions.
  template <typename T, int N>
  std::vector<T> refine_peaks(const Array<T, N, 1>& orientation_histogram,
                              const std::vector<int>& ori)
  {
    std::vector<T> oriT(ori.size());
    for (size_t i = 0; i < ori.size(); ++i)
      oriT[i] = refine_peak<T, N>(orientation_histogram, ori[i]);
    return oriT;
  }

  //! @brief Helper class.
  class DO_SARA_EXPORT ComputeDominantOrientations
  {
  public:
    ComputeDominantOrientations(float peak_ratio_thres = 0.8f,
                                float patch_truncation_factor = 3.f,
                                float blur_factor = 1.5f);

    std::vector<float> operator()(const ImageView<Vector2f>& gradients,
                                  float x, float y, float sigma) const;

    std::vector<float> operator()(const ImagePyramid<Vector2f>& pyramid,
                                  const OERegion& extremum,
                                  const Point2i& scale_octave_pair) const;

    void operator()(const ImagePyramid<Vector2f>& pyramid,
                    std::vector<OERegion>& extrema,
                    std::vector<Point2i>& scale_octave_pairs) const;

  private:
    float _peak_ratio_thres;
    float _patch_truncation_factor;
    float _blur_factor;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
