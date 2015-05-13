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

namespace DO {

  void drawScaleSpaceExtremum(const ImagePyramid<float>& I,
                              float x, float y, float s,
                              int o, const Rgb8& c)
  {
    float z = I.octaveScalingFactor(o);
    x *= z;
    y *= z;
    s *= z;
    if (s < 3.f)
      s = 3.f;
    drawCircle(Point2f(x,y),s,c,3);
  }

  void drawExtrema(const ImagePyramid<float>& pyramid,
                   const vector<OERegion>& extrema,
                   int s, int o, bool rescaleColor)
  {
    if (rescaleColor)
      display(colorRescale(pyramid(s,o)), 0, 0, pyramid.octaveScalingFactor(o));
    else
      display(pyramid(s,o), 0, 0, pyramid.octaveScalingFactor(o));
    for (size_t i = 0; i != extrema.size(); ++i)
    {
      drawScaleSpaceExtremum(
        pyramid,
        extrema[i].x(), extrema[i].y(), extrema[i].scale(),
        o, extrema[i].extremumType() == OERegion::Max?Red8:Blue8);
    }
  }

  void highlightPatch(const ImagePyramid<float>& D,
                      float x, float y, float s, int o)
  {
    const float magFactor = 3.f;
    float z = D.octaveScalingFactor(o);
    int r = static_cast<int>(floor(1.5f*s*magFactor*z + 0.5f));
    x *= z;
    y *= z;
    drawRect(intRound(x)-r, intRound(y)-r, 2*r+1, 2*r+1, Green8, 3);
  }

  void checkPatch(const Image<float>& I, int x, int y, int w, int h,
                  double fact)
  {
    Image<float> patch( getImagePatch(I,x,y,w,h) );
    setActiveWindow( openWindow(w*fact, h*fact, "Check image patch") );
    display(colorRescale(patch), 0, 0, fact);
    getKey();
    closeWindow();
  }

  void checkPatch(const Image<float>& I, float x, float y, float s, double fact)
  {
    const float magFactor = 3.f;
    int r = int(s*1.5f*magFactor + 0.5f);
    int w = 2*r+1;
    int h = 2*r+1;
    checkPatch(I, intRound(x)-r, intRound(y)-r, w, h, fact);
  }

} /* namespace DO */