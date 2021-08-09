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

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Features.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup FeatureDetectors
    @defgroup UtilitiesDebug Visual Inspection Utilities
    @{
   */

  //! @brief Visually inspect the image pyramid.
  template <typename T>
  void display_image_pyramid(const ImagePyramid<T>& pyramid,
                             bool rescale = false)
  {
    using namespace std;
    for (int o = 0; o < pyramid.num_octaves(); ++o)
    {
      cout << "Octave " << o << endl;
      cout << "- scaling factor = " << pyramid.octave_scaling_factor(o) << endl;
      for (int s = 0; s != int(pyramid(o).size()); ++s)
      {
        cout << "Image " << s << endl;
        cout << "Image relative scale to octave = "
             << pyramid.scale_relative_to_octave(s) << endl;

        display(rescale ? color_rescale(pyramid(s, o)) : pyramid(s, o), 0, 0,
                pyramid.octave_scaling_factor(o));
        get_key();
      }
    }
  }

  //! @{
  //! @brief Visually inspect local extrema.
  DO_SARA_EXPORT
  void draw_scale_space_extremum(const ImagePyramid<float>& I, float x, float y,
                                 float s, int o, const Rgb8& c);

  DO_SARA_EXPORT
  void draw_extrema(const ImagePyramid<float>& pyramid,
                    const std::vector<OERegion>& extrema, int s, int o,
                    bool rescale_color = true);

  DO_SARA_EXPORT
  void highlight_patch(const ImagePyramid<float>& D, float x, float y, float s,
                       int o);

  DO_SARA_EXPORT
  void check_patch(const ImageView<float>& I, int x, int y, int w, int h,
                   double fact = 50.);

  DO_SARA_EXPORT
  void check_patch(const ImageView<float>& I, float x, float y, float s,
                   double fact = 20.);
  //! @}

  //! @brief Visually inspect the descriptor.
  template <typename T, int N>
  void view_histogram(const Array<T, N, 1>& histogram)
  {
    using namespace std;
    set_active_window(create_window(720, 200, "Histogram"));
    auto w = int_round(720. / histogram.size());
    float max = histogram.maxCoeff();
    for (int i = 0; i < histogram.size(); ++i)
    {
      auto h = int_round(histogram(i) / max * 200);
      fill_rect(i * w, 200 - h, w, h, Blue8);
    }
    cout << histogram.transpose() << endl;
    get_key();
    close_window();
  }

  //! Check the grid on which we are drawing.
  template <int N>
  void draw_sift_grid(float x, float y, float sigma, float theta,
                      float octave_scale_factor, int pen_width = 1)
  {
    const auto lambda = 3.f;
    const auto l = lambda * sigma;
    Eigen::Vector2f grid[N + 1][N + 1];

    auto T = Matrix2f{};
    theta = 0;
    // clang-format off
    T << cos(theta), -sin(theta),
         sin(theta),  cos(theta);
    // clang-format on
    T *= l;

    for (auto v = 0; v < N + 1; ++v)
      for (auto u = 0; u < N + 1; ++u)
        grid[u][v] =
            (Eigen::Vector2f{x, y} + T * Vector2f{u - N / 2.f, v - N / 2.f}) *
            octave_scale_factor;
    for (auto i = 0; i < N + 1; ++i)
      draw_line(grid[0][i], grid[N][i], Green8, pen_width);
    for (auto i = 0; i < N + 1; ++i)
      draw_line(grid[i][0], grid[i][N], Green8, pen_width);

    auto a = Vector2f{x, y};
    a *= octave_scale_factor;
    auto b = Vector2f{};
    b = a + octave_scale_factor * N / 2.f * T * Vector2f{1.f, 0.f};
    draw_line(a, b, Red8, pen_width + 2);
  }

  //! @}

}}  // namespace DO::Sara
