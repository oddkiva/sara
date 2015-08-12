// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_FEATUREDESCRIPTORS_ORIENTATION_HPP
#define DO_SARA_FEATUREDESCRIPTORS_ORIENTATION_HPP

#include <DO/Sara/Defines.hpp>


namespace DO { namespace Sara {

  /*!
    \ingroup FeatureDesriptors
    \defgroup Orientation Dominant Orientation Assignment
    @{
  */

  /*!
    \brief Computes the image gradients of image \f$I\f$ in polar coordinates.

    In each pixel \f$(x,y) \in [0,w[ \times [0,h[\f$,
    \f$\nabla I(x,y)\f$ is the 2D vector \f$(r,\theta)\f$ where:
    - \f$r = 2 ||\nabla I(x,y)||\f$,
    - \f$\theta = \mathrm{angle}( \nabla I(x,y) )\f$.
   */
  template <typename T>
  Image<Matrix<T,2,1>> gradient_polar_coordinates(const Image<T>& f)
  {
    Image<Matrix<T, 2, 1>> nabla_f{ gradient(f) };
    for (auto it = nabla_f.begin(); it != nabla_f.end(); ++it)
    {
      auto r = 2*it->norm();
      auto theta = atan2(it->y(), it->x());
      *it << r, theta;
    }
    return nabla_f;
  }

  /*!
    \brief Computes the image gradients in polar coordinates for each image in
    the pyramid.
   */
  template <typename T>
  ImagePyramid<Matrix<T, 2, 1>> gradient_polar_coordinates(const ImagePyramid<T>& pyramid)
  {
    ImagePyramid<Matrix<T, 2, 1> > gradient_pyramid;
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

  /*!
    \brief  Computes the orientation histogram on a local patch around keypoint
    \f$(x,y,\sigma)\f$.
   */
  template <typename T, int N>
  void compute_orientation_histogram(Array<T, N, 1>& orientation_histogram,
                                     const Image<Matrix<T,2,1>>& gradient_polar,
                                     T x, T y, T s,
                                     T patch_truncation_factor = T(3),
                                     T blur_factor = T(1.5))
  {
    // Weighted histogram of gradients.
    orientation_histogram.setZero();

    // Rounding of the coordinates.
    int xi = int_round(x);
    int yi = int_round(y);

    // std deviation of the gaussian weight (cf. [Lowe, IJCV 2004])
    T sigma = s*blur_factor;

    // Patch radius on which the histogram of gradients is performed.
    int patch_radius = int_round(sigma*patch_truncation_factor);

    // Accumulate the histogram of orientations.
    for (int v = -patch_radius; v <= patch_radius; ++v)
    {
      for (int u = -patch_radius; u <= patch_radius; ++u)
      {
        if ( xi+u < 0 || xi+u >= gradient_polar.width()  ||
             yi+v < 0 || yi+v >= gradient_polar.height() )
          continue;

        T mag = gradient_polar(xi+u, yi+v)(0);
        T ori = gradient_polar(xi+u, yi+v)(1);
        // ori is in \f$]-\pi, \pi]\f$, so translate ori by \f$2*\pi\f$ if it is
        // negative.
#ifndef LOWE
        ori = ori < 0 ? ori+T(2.*M_PI) : ori;
        int bin_index = floor(ori/T(2*M_PI) * T(N));
        bin_index %= N;
#else
        int bin_index = int( (N * (ori + M_PI + 0.001f) / (2.0f * M_PI)) );
        bin_index = std::min(bin_index, N - 1);
#endif
        if (bin_index < 0 || bin_index >= N)
        {
          std::ostringstream oss;
          oss << "Orientation bin index out of range: " << bin_index
              << " theta = " << ori << std::endl;
          std::cerr << oss.str() << std::endl;
          throw std::out_of_range(oss.str());
        }

        // Give more emphasis to gradient orientations that lie closer to the
        // keypoint location.
        T weight = exp(-(u*u+v*v)/(T(2)*sigma*sigma));
        // Also give more emphasis to gradient with large magnitude.
        orientation_histogram( bin_index ) += weight*mag;
      }
    }
  }

  /*!
    \brief This is used in [Lowe, IJCV 2004] to determine keypoint orientations.

    Basically, the histogram is convolved 6 times with the mean kernel
    \f$[1/3, 1/3, 1/3]\f$.
   */
  template <typename T, int N>
  void lowe_smooth_histogram(Array<T, N, 1>& orientation_histogram)
  {
    for (int iter = 0; iter < 6; ++iter)
    {
      float first = orientation_histogram(0);
      float prev = orientation_histogram(N-1);
      for (int i = 0; i < N-1; ++i)
      {
        float val = (prev+orientation_histogram(i)+orientation_histogram(i+1))/3.f;
        prev = orientation_histogram(i);
        orientation_histogram(i) = val;
      }
      orientation_histogram(N-1) = (prev+orientation_histogram(N-1)+first)/3.f;
    }
  }

  /*!
    \brief  This is used in [Lowe, IJCV 2004] to find histogram peaks.

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
      if ( orientation_histogram(i) >= peak_ratio_thres*max &&
           orientation_histogram(i) > orientation_histogram((i-1+N)%N)     &&
           orientation_histogram(i) > orientation_histogram((i+1)%N)       )
        orientation_peaks.push_back(i);
    return orientation_peaks;
  }

  /*!
    \brief Refine peaks as in [Lowe, IJCV 2004] by interpolation based on a
    second-order Taylor approximation.
   */
  template <typename T, int N>
  T refine_peak(const Array<T, N, 1>& orientation_histogram, int i)
  {
    T y0 = orientation_histogram( (i-1+N) % N );
    T y1 = orientation_histogram( i );
    T y2 = orientation_histogram( (i+1) % N );
    // Denote the orientation histogram function by \f$f\f$.
    // perform a 2nd-order Taylor approximation:
    // \f$f(x+h) = f(x) + f'(x)h + f''(x) h^2/2\f$
    // We approximate \f$f'\f$ and \f$f''\f$ by finite difference.
    T fprime = (y2-y0) / 2.f;
    T fsecond = y0 - 2.f*y1 + y2;
    // Maximize w.r.t. to \f$h\f$, derive the expression.
    // Thus \f$h = -f'(x)/f''(x)\f$.
    T h = -fprime / fsecond;
    // Add the offset \f$h\f$ to get the refined orientation value.
    // Note that we also add the 0.5f offset, because samples are assumed taken
    // on the middle of the interval \f$[i, i+1)\f$.
    return T(i)+T(0.5)+h;
  }

  //! \brief Helper functions.
  template <typename T, int N>
  std::vector<T> refine_peaks(const Array<T, N, 1>& orientation_histogram,
                              const std::vector<int>& ori)
  {
    std::vector<T> oriT(ori.size());
    for (size_t i = 0; i < ori.size(); ++i)
      oriT[i] = refine_peak<T,N>(orientation_histogram, ori[i]);
    return oriT;
  }

  //! \brief Basically a helper class.
  class DO_EXPORT ComputeDominantOrientations
  {
  public:
    ComputeDominantOrientations(float peak_ratio_thres = 0.8f,
                                float patch_truncation_factor = 3.f,
                                float blur_factor = 1.5f);

    std::vector<float> operator()(const Image<Vector2f>& gradients,
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


#endif /* DO_SARA_FEATUREDESCRIPTORS_ORIENTATION_HPP */