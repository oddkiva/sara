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


using namespace std;


namespace DO { namespace Sara {

  ComputeDominantOrientations::
  ComputeDominantOrientations(float peak_ratio_thres,
                              float patch_truncation_factor,
                              float blur_factor)
    : _peak_ratio_thres(peak_ratio_thres)
    , _patch_truncation_factor(patch_truncation_factor)
    , _blur_factor(blur_factor)
  {
  }

  std::vector<float>
  ComputeDominantOrientations::
  operator()(const ImageView<Vector2f>& gradients,
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

  std::vector<float>
  ComputeDominantOrientations::
  operator()(const ImagePyramid<Vector2f>& pyramid,
             const OERegion& extremum,
             const Point2i& scale_octave_pair) const
  {
    const auto& s_index = scale_octave_pair(0);
    const auto& o_index = scale_octave_pair(1);
    auto x = extremum.x();
    auto y = extremum.y();
    auto s = pyramid.scale_relative_to_octave(s_index);
    return this->operator()(pyramid(s_index, o_index), x, y, s);
  }

  void
  ComputeDominantOrientations::
  operator()(const ImagePyramid<Vector2f>& pyramid,
             vector<OERegion>& extrema,
             vector<Point2i>& scale_octave_pairs) const
  {
    using namespace std;

    auto e2 = vector<OERegion>{};
    auto so2 = vector<Point2i>{};

    e2.reserve(extrema.size() * 2);
    so2.reserve(extrema.size() * 2);

    for (size_t i = 0; i != extrema.size(); ++i)
    {
      auto orientations = vector<float>{};
      orientations = this->operator()(pyramid, extrema[i], scale_octave_pairs[i]);
      for (size_t o = 0; o != orientations.size(); ++o)
      {
        // Recopy.
        so2.push_back(scale_octave_pairs[i]);
        e2.push_back(extrema[i]);
        // Assign dominant orientation.
        e2.back().orientation = orientations[o];
      }
    }

    e2.swap(extrema);
    so2.swap(scale_octave_pairs);
  }

} /* namespace Sara */
} /* namespace DO */
