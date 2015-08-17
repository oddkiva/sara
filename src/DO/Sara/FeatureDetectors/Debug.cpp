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

#include <DO/Sara/Geometry/Tools/Utilities.hpp>


using namespace std;


namespace DO { namespace Sara {

  void draw_scale_space_extremum(const ImagePyramid<float>& I,
                                 float x, float y, float s,
                                 int o, const Rgb8& c)
  {
    float z = I.octave_scaling_factor(o);
    x *= z;
    y *= z;
    s *= z;
    if (s < 3.f)
      s = 3.f;
    draw_circle(Point2f(x,y),s,c,3);
  }

  void draw_extrema(const ImagePyramid<float>& pyramid,
                    const vector<OERegion>& extrema,
                    int s, int o, bool rescale_color)
  {
    if (rescale_color)
      display(color_rescale(pyramid(s,o)), 0, 0, pyramid.octave_scaling_factor(o));
    else
      display(pyramid(s,o), 0, 0, pyramid.octave_scaling_factor(o));

    for (size_t i = 0; i != extrema.size(); ++i)
    {
      draw_scale_space_extremum(
        pyramid,
        extrema[i].x(), extrema[i].y(), extrema[i].scale(),
        o, extrema[i].extremum_type() == OERegion::Max?Red8:Blue8);
    }
  }

  void highlight_patch(const ImagePyramid<float>& D,
                       float x, float y, float s, int o)
  {
    const float magnitude_factor = 3.f;
    float z = D.octave_scaling_factor(o);
    int r = static_cast<int>(floor(1.5f*s*magnitude_factor*z + 0.5f));
    x *= z;
    y *= z;
    draw_rect(int_round(x)-r, int_round(y)-r, 2*r+1, 2*r+1, Green8, 3);
  }

  void check_patch(const Image<float>& I, int x, int y, int w, int h,
                   double fact)
  {
    auto patch = get_subimage(I,x,y,w,h);
    auto window = create_window(int_round(w*fact), int_round(h*fact), "Check image patch");
    set_active_window(window);
    display(color_rescale(patch), 0, 0, fact);
    get_key();
    close_window();
  }

  void check_patch(const Image<float>& I, float x, float y, float s, double fact)
  {
    const auto scaling_factor = 3.f;
    auto r = s * 1.5f * scaling_factor;
    int w = 2 * int_round(r) + 1;
    int h = 2 * int_round(r) + 1;

    check_patch(I, int_round(x - r), int_round(y - r), w, h, fact);
  }

} /* namespace Sara */
} /* namespace DO */
