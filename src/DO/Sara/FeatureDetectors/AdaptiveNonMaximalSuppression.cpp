// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/FeatureDetectors.hpp>


using namespace std;


namespace DO { namespace Sara {

  // Greater comparison functor for the adaptive non maximal suppression
  // algorithm.
  typedef pair<size_t, float> IndexScore;

  vector<pair<size_t, float>>
  adaptive_non_maximal_suppression(const vector<OERegion>& features,
                                float c_robust)
  {
    auto compare_index_score = [](const IndexScore& a, const IndexScore& b) {
      return a.second > b.second;
    };

    // Create the ordered list sorted by decreasing strength.
    auto idx_score_pairs = vector<IndexScore>{ features.size() };

    // Notice that I readily multiply the scores by $c_\textrm{robust}$.
    for (size_t i = 0; i != features.size(); ++i)
      idx_score_pairs[i] = make_pair(i, c_robust*features[i].extremum_value());

    // Sort features by decreasing strength.
    sort(
      idx_score_pairs.begin(), idx_score_pairs.end(),
      compare_index_score); // (O(N * log N)

    // Compute the suppression radius.
    vector<IndexScore> idx_sq_radius_pairs(features.size());
    const auto infty = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i != idx_score_pairs.size(); ++i)
    {
      // Start from infinite (squared) radius.
      float squared_radius = infty;
      if (i == 0)
      {
        idx_sq_radius_pairs[i] = make_pair(idx_score_pairs[i].first, squared_radius);
        continue;
      }

      // f(x_0) >= f(x_1) >= ... >= f(x_i) >= ... >= f(x_{N-1})
      // So:
      //  $f(x_i) > c f(x_{i+1}$
      //  $f(x_i) > c f(x_{i+2}$
      //   ...
      //  $f(x_i) > c f(x_{N-1}$
      //
      // We only care about stronger interest points and we have to check:
      //  (A_0)     <=> f(x_i) < c f(x_{0}) ?
      //  (A_1)     <=> f(x_i) < c f(x_{1}) ?
      //   ...
      //  (A_{i-1}) <=> f(x_i) < c f(x_{i-1}) ?
      //
      // If all (A_i) is false, return an infinite radius.
      auto it = lower_bound(
        idx_sq_radius_pairs.begin(), idx_sq_radius_pairs.begin() + i,
        idx_score_pairs[i], compare_index_score); // O(log i)

      if (it != idx_sq_radius_pairs.end())
      {
        size_t sz = it-idx_sq_radius_pairs.begin();
        for (size_t j = 1; j != sz; ++j)
        {
          const Point2f& xi = features[idx_score_pairs[i].first].center();
          const Point2f& xj = features[idx_sq_radius_pairs[j].first].center();
          float rr = (xi - xj).squaredNorm();
          if (rr < squared_radius)
            squared_radius = rr;
        } // O(i)
      }

      idx_sq_radius_pairs[i] = make_pair(idx_score_pairs[i].first, squared_radius);
    }

    sort(
      idx_sq_radius_pairs.begin(), idx_sq_radius_pairs.end(),
      compare_index_score);

    return idx_sq_radius_pairs;
  }


} /* namespace Sara */
} /* namespace DO */