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

#include <Eigen/StdVector>

#include <DO/Sara/FeatureDescriptors.hpp>

#include <DO/Sara/ImageProcessing/Differential.hpp>


using namespace std;


namespace DO { namespace Sara {

  auto gradient_polar_coordinates(const ImageView<float>& f)
      -> Image<Eigen::Vector2f>
  {
    // The impact on SIFT computation is non negligible:
    //
    // Typical timing of the naive CPU implementation on a 1080p video:
    // [compute_sift_keypoints:56] gradient of Gaussian computation time = 307.9 ms
    //
    // Typical timing of Halide CPU implementation on a 1080p video:
    // [compute_sift_keypoints:56] gradient of Gaussian computation time = 82.8392 ms

    // TODO: find out why Halide crashes for small image height...
#ifdef DO_SARA_USE_HALIDE
    auto mag = Image<float>{f.sizes()};
    auto ori = Image<float>{f.sizes()};
    gradient_in_polar_coordinates(f, mag, ori);

    auto nabla_f = Image<Eigen::Vector2f>{f.sizes()};
    std::transform(mag.begin(), mag.end(), ori.begin(), nabla_f.begin(),
                   [](float m, float o) -> Eigen::Vector2f {
                     return {m, o};
                   });
#else
    auto nabla_f = gradient(f);
    std::transform(nabla_f.begin(), nabla_f.end(), nabla_f.begin(),
                   [](const auto& g) -> Eigen::Vector2f {
                     const auto r = 2 * g.norm();
                     const auto theta = std::atan2(g.y(), g.x());
                     return {r, theta};
                   });
#endif
    return nabla_f;
  }

  auto gradient_polar_coordinates(const ImagePyramid<float>& src,
                                  ImagePyramid<float>& gradient_magnitude,
                                  ImagePyramid<float>& gradient_orientation)
      -> void
  {
    gradient_magnitude.reset(src.octave_count(), src.scale_count_per_octave(),
                             src.scale_initial(), src.scale_geometric_factor());
    gradient_orientation.reset(src.octave_count(), src.scale_count_per_octave(),
                               src.scale_initial(),
                               src.scale_geometric_factor());

    for (int o = 0; o < src.octave_count(); ++o)
    {
      gradient_magnitude.octave_scaling_factor(o) =
          src.octave_scaling_factor(o);
      gradient_orientation.octave_scaling_factor(o) =
          src.octave_scaling_factor(o);
      for (int s = 0; s < src.scale_count_per_octave(); ++s)
        gradient_in_polar_coordinates(src(s, o),  //
                                      gradient_magnitude(s, o),
                                      gradient_orientation(s, o));
    }
  }

  ComputeDominantOrientations::ComputeDominantOrientations(
      float peak_ratio_thres, float patch_truncation_factor, float blur_factor)
    : _peak_ratio_thres(peak_ratio_thres)
    , _patch_truncation_factor(patch_truncation_factor)
    , _blur_factor(blur_factor)
  {
  }

  std::vector<float>
  ComputeDominantOrientations::operator()(const ImageView<Vector2f>& gradients,
                                          float x, float y, float sigma) const
  {
    // Compute histogram of gradients as in [Lowe, IJCV 2004].
    constexpr auto O = 36;
    auto orientation_histogram = Array<float, O, 1>{};
    compute_orientation_histogram(orientation_histogram, gradients, x, y, sigma,
                                  _patch_truncation_factor, _blur_factor);

    // Smooth histogram as in the initial implementation of [Lowe, IJCV 2004].
    lowe_smooth_histogram(orientation_histogram);
    auto peak_indices = find_peaks(orientation_histogram, _peak_ratio_thres);

    // Refine peaks as in [Lowe, IJCV 2004].
    auto peaks = refine_peaks(orientation_histogram, peak_indices);

    // Convert orientation to radian.
    for (size_t i = 0; i != peaks.size(); ++i)
    {
      // Orientations in $[0, 2\pi[$
      peaks[i] *= static_cast<float>(2 * M_PI) / O;
      // Orientations in $[-\pi, \pi[$
      if (peaks[i] > float(M_PI))
        peaks[i] -= 2.f * float(M_PI);
    }

    return peaks;
  }

  std::vector<float> ComputeDominantOrientations::operator()(
      const ImagePyramid<Vector2f>& pyramid, const OERegion& extremum,
      const Point2i& scale_octave_pair) const
  {
    const auto& s_index = scale_octave_pair(0);
    const auto& o_index = scale_octave_pair(1);
    const auto x = extremum.x();
    const auto y = extremum.y();
    const auto s =
        static_cast<float>(pyramid.scale_relative_to_octave(s_index));
    return this->operator()(pyramid(s_index, o_index), x, y, s);
  }

  void ComputeDominantOrientations::operator()(
      const ImagePyramid<Vector2f>& pyramid, vector<OERegion>& extrema,
      vector<Point2i>& scale_octave_pairs) const
  {
    using namespace std;

    auto e2 = vector<OERegion>{};
    auto so2 = vector<Point2i>{};

    e2.reserve(extrema.size() * 2);
    so2.reserve(extrema.size() * 2);

    // FIXME: parallelize this.
    for (size_t i = 0; i != extrema.size(); ++i)
    {
      auto orientations = vector<float>{};
      orientations =
          this->operator()(pyramid, extrema[i], scale_octave_pairs[i]);

      {
        for (size_t o = 0; o != orientations.size(); ++o)
        {
          // Recopy.
          so2.push_back(scale_octave_pairs[i]);
          e2.push_back(extrema[i]);
          // Assign dominant orientation.
          e2.back().orientation = orientations[o];
        }
      }
    }

    e2.swap(extrema);
    so2.swap(scale_octave_pairs);
  }

}}  // namespace DO::Sara
