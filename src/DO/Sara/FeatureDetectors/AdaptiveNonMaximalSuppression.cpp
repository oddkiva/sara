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

#include <DO/Sara/FeatureDetectors.hpp>

using namespace std;

namespace DO { namespace Sara {

  // Greater comparison functor for the adaptive non maximal suppression
  // algorithm.
  typedef pair<size_t, float> IndexScore;
  struct CompareIndexScore {
    bool operator()(const IndexScore& lhs, const IndexScore& rhs) const
    { return lhs.second > rhs.second; }
  };

  vector<pair<size_t, float> >
  adaptiveNonMaximalSuppression(const vector<OERegion>& features,
                                float c_robust)
  {
    // Create the ordered list sorted by decreasing strength.
    vector<IndexScore> is(features.size()); // is = 'index-strength pairs'.
    // Notice that I readily multiply the scores by $c_\textrm{robust}$.
    for (size_t i = 0; i != features.size(); ++i)
      is[i] = make_pair(i, c_robust*features[i].extremumValue());
    // Sort features by decreasing strength.
    CompareIndexScore cmp;
    sort(is.begin(), is.end(), cmp); // O(N log(N) )

    // Compute the suppression radius.
    vector<IndexScore> ir(features.size()); // ir = 'index radius size';
    const float infty = std::numeric_limits<float>::infinity(); // shortcut.
    for (size_t i = 0; i != is.size(); ++i)
    {
      // Start from infinite (squared) radius.
      float r = infty;
      if (i == 0)
      {
        ir[i] = make_pair(is[i].first, r);
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
      vector<IndexScore>::iterator
        it = lower_bound(ir.begin(), ir.begin()+i, is[i], cmp); // O(log i)
      if (it != ir.end())
      {
        size_t sz = it-ir.begin();
        for (size_t j = 1; j != sz; ++j)
        {
          const Point2f& xi = features[is[i].first].center();
          const Point2f& xj = features[ir[j].first].center();
          float rr = (xi - xj).squaredNorm();
          if (rr < r)
            r = rr;
        } // O(i)
      }
      ir[i] = make_pair(is[i].first, r);
    }

    sort(ir.begin(), ir.end(), cmp);
    return ir;
  }


} /* namespace Sara */
} /* namespace DO */
