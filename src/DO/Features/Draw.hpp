// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_FEATURES_DRAW_HPP
#define DO_FEATURES_DRAW_HPP

namespace DO {

  template <typename Key>
  void drawKeypoints(const std::vector<Key>& keypoints, const Color3ub& c,
                     float scale = 1.0f, const Point2f& off = Point2f::Zero())
  {
    for (size_t i = 0; i < keypoints.size(); ++i)
      keypoints[i].feat().drawOnScreen(c, scale, off);
  }

} /* namespace DO */

#endif /* DO_AFFINECOVARIANTFEATURES_DRAW_H */
