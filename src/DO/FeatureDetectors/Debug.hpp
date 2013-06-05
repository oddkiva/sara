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

//! @file

#ifndef DO_FEATUREDETECTORS_DEBUG_HPP
#define DO_FEATUREDETECTORS_DEBUG_HPP

namespace DO {

  /*!
    \ingroup FeatureDetectors
    \defgroup UtilitiesDebug Utilities and Debug
    @{
   */

  template <typename T>
  inline int intRound(T x) { return static_cast<int>(floor(x+T(0.5))); }

  // Check the image pyramid.
  template <typename T>
  void displayImagePyramid(const ImagePyramid<T>& pyramid,
                           bool rescale = false)
  {
    using namespace std;
    for (int o = 0; o < pyramid.numOctaves(); ++o)
    {
      cout << "Octave " << o << endl;
      cout << "- scaling factor = "
           << pyramid.octaveScalingFactor(o) << endl;
      for (int s = 0; s != int(pyramid(o).size()); ++s)
      {
        cout << "Image " << s << endl;
        cout << "Image relative scale to octave = "
             << pyramid.octRelScale(s) << endl;

        display(rescale ? colorRescale(pyramid(s,o)) : pyramid(s,o), 
          0, 0, pyramid.octaveScalingFactor(o));
        getKey();
      }
    }
  }

  // Check the local extrema.
  void drawScaleSpaceExtremum(const ImagePyramid<float>& I,
                              float x, float y, float s,
                              int o, const Rgb8& c);

  void drawExtrema(const ImagePyramid<float>& pyramid,
                   const std::vector<OERegion>& extrema,
                   int s, int o, bool rescaleColor = true);

  void highlightPatch(const ImagePyramid<float>& D,
                      float x, float y, float s, int o);

  void checkPatch(const Image<float>& I, int x, int y, int w, int h,
                  double fact = 50.);

  void checkPatch(const Image<float>& I,float x, float y, float s,
                  double fact = 20.);

  template <typename T, int N>
  void viewHistogram(const Array<T, N, 1>& histogram)
  {
    using namespace std;
    setActiveWindow(openWindow(720, 200, "Histogram"));
    int w = 720./histogram.size();
    float max = histogram.maxCoeff();
    for (int i = 0; i < histogram.size(); ++i)
    {
      int h = histogram(i)/max*200;
      fillRect(i*w, 200-h, w, h, Blue8);
    }
    cout << histogram.transpose() << endl;
    getKey();
    closeWindow();
  }

  //! @}

} /* namespace DO */

#endif /* DO_FEATUREDETECTORS_DEBUG_HPP */
