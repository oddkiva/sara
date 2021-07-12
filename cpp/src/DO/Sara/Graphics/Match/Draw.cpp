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

#include <fstream>
#include <sstream>
#include <string>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Match.hpp>


using namespace std;


namespace DO::Sara {

  void draw_image_pair(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                       const Point2f& off2, float scale)
  {
    display(I1, Point2i::Zero(), scale);
    display(I2, (off2 * scale).cast<int>(), scale);
  }

  void draw_match(const Match& m, const Color3ub& c, const Point2f& t, float z)
  {
    draw(m.x(), c, z);
    draw(m.y(), c, z, t);
    Point2f p1 = m.x_pos() * z;
    Point2f p2 = (m.y_pos() + t) * z;
    draw_line(p1, p2, c);
  }

  void draw_matches(const vector<Match>& matches, const Point2f& off2, float z)
  {
    for (auto m = matches.begin(); m != matches.end(); ++m)
      draw_match(*m, Color3ub(rand() % 256, rand() % 256, rand() % 256), off2,
                 z);
  }

  void check_matches(const Image<Rgb8>& I1, const Image<Rgb8>& I2,
                     const vector<Match>& matches, bool redraw_everytime,
                     float z)
  {
    Point2f off{float(I1.width()), 0.f};
    draw_image_pair(I1, I2);
    for (auto m = matches.begin(); m != matches.end(); ++m)
    {
      if (redraw_everytime)
        draw_image_pair(I1, I2, z);
      draw_match(*m, Color3ub(rand() % 256, rand() % 256, rand() % 256), off,
                 z);
      cout << *m << endl;
      get_key();
    }
  }

}  // namespace DO::Sara
