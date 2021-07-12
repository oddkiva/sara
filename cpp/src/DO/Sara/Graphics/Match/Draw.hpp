// ========================================================================== //
// This file is part of Sara", a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Match.hpp>


namespace DO::Sara {

  //! @{
  //! View matches.
  DO_SARA_EXPORT
  void draw_image_pair(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                       const Point2f& off2, float scale = 1.0f);

  inline void draw_image_pair(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                              float scale = 1.0f)
  {
    draw_image_pair(I1, I2, Point2f(I1.width() * scale, 0.f), scale);
  }

  inline void draw_image_pair_v(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                                float scale = 1.0f)
  {
    draw_image_pair(I1, I2, Point2f(0.f, I1.height() * scale), scale);
  }

  DO_SARA_EXPORT
  void draw_match(const Match& m, const Color3ub& c, const Point2f& off2,
                  float z = 1.f);

  DO_SARA_EXPORT
  void draw_matches(const std::vector<Match>& matches, const Point2f& off2,
                    float z = 1.f);

  DO_SARA_EXPORT
  void check_matches(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                     const std::vector<Match>& matches,
                     bool redraw_everytime = false, float z = 1.f);
  //! @}

}  // namespace DO::Sara
