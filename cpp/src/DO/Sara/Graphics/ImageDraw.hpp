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


class QImage;


namespace DO { namespace Sara {

  /*!
   *  @ingroup Graphics
   *  @defgroup ImageDrawing Drawing Image
   *
   *  @{
   */

  /*!
   *  @brief Draw point on image.
   *  @param[in]  image image.
   *  @param[in]  x,y   coordinates.
   *  @param[in]  c     RGB color.
   */
  DO_SARA_EXPORT
  auto draw_point(ImageView<Rgb8>& image, int x, int y, const Rgb8& c) -> void;

  /*!
   *  @brief Draw line on image.
   *  @param[in]  image       image.
   *  @param[in]  x1,y1,x2,y2 start and end points of the line.
   *  @param[in]  c           RGB color.
   *  @param[in]  pen_width    width of the contour.
   */
  DO_SARA_EXPORT
  auto draw_line(ImageView<Rgb8>& image, int x1, int y1, int x2, int y2,
                 const Rgb8& c, int pen_width = 1, bool antialiasing = true)
      -> void;

  inline auto draw_line(ImageView<Rgb8>& image, const Eigen::Vector2f& a,
                        const Eigen::Vector2f& b, const Rgb8& c,
                        int pen_width = 1, bool antialiasing = true) -> void
  {
    auto round = [](float x) { return static_cast<int>(x + 0.5f); };
    draw_line(image, round(a.x()), round(a.y()), round(b.x()), round(b.y()), c,
              pen_width, antialiasing);
  }

  /*!
   *  @brief Draw rectangle on image.
   *  @param[in]  image     image.
   *  @param[in]  x,y,w,h   start and end points of the line.
   *  @param[in]  c         RGB color.
   *  @param[in]  pen_width  width of the contour.
   */
  DO_SARA_EXPORT auto draw_rect(ImageView<Rgb8>& image, int x, int y, int w,
                                int h, const Rgb8& c, int pen_width = 1)
      -> void;

  /*!
   *  @brief Draw circle on image.
   *  @param[in]  image    image.
   *  @param[in]  xc,yc    circle center.
   *  @param[in]  r        circle radius.
   *  @param[in]  c        RGB color.
   *  @param[in]  pen_width width of the contour.
   */
  DO_SARA_EXPORT
  auto draw_circle(ImageView<Rgb8>& image, int xc, int yc, int r, const Rgb8& c,
                   int pen_width = 1, bool antialiasing = true) -> void;

  inline auto draw_circle(ImageView<Rgb8>& image, const Eigen::Vector2f& c,
                          int r, const Rgb8& color, int pen_width = 1,
                          bool antialiasing = true) -> void
  {
    auto round = [](float x) { return static_cast<int>(x + 0.5f); };
    draw_circle(image, round(c.x()), round(c.y()), r, color, pen_width,
                antialiasing);
  }

  //! @brief Draw ellipse.
  auto draw_ellipse(ImageView<Rgb8>& image, const Eigen::Vector2f& center,
                    float r1, float r2, float degree, const Rgb8& c,
                    int pen_width = 1, bool antialiasing = true) -> void;

  //! @brief Draw arrow.
  DO_SARA_EXPORT
  auto draw_arrow(ImageView<Rgb8>& image, int x1, int y1, int x2, int y2,
                  const Rgb8& col, int pen_width = 1, int arrow_width = 8,
                  int arrow_height = 5, int style = 0, bool antialiasing = true)
      -> void;

  inline auto draw_arrow(ImageView<Rgb8>& image, const Eigen::Vector2f& a,
                         const Eigen::Vector2f& b, const Rgb8& col,
                         int pen_width = 1, int arrow_width = 8,
                         int arrow_height = 5, int style = 0,
                         bool antialiasing = true) -> void
  {
    static constexpr auto round = [](const float x) {
      return static_cast<int>(std::round(x));
    };
    draw_arrow(image, round(a.x()), round(a.y()), round(b.x()), round(b.y()),
               col, pen_width, arrow_width, arrow_height, style, antialiasing);
  }

  //! @brief Draw text.
  DO_SARA_EXPORT
  auto draw_text(ImageView<Rgb8>& image, int x, int y,  //
                 const std::string& text, const Rgb8& color = White8,
                 int font_size = 10, float orientation = 0.f,
                 bool italic = false, bool bold = false, bool underline = false,
                 int pen_width = 1, bool antialiasing = true) -> void;

  /*!
   *  @brief Draw color-filled rectangle on image.
   *  @param[in]  image     image.
   *  @param[in]  x,y,w,h   start and end points of the line.
   *  @param[in]  c         RGB color.
   */
  DO_SARA_EXPORT
  auto fill_rect(ImageView<Rgb8>& image, int x, int y, int w, int h,
                 const Rgb8& c) -> void;

  /*!
   *  @brief Draw color-filled circle on image.
   *  @param[in]  image  image.
   *  @param[in]  x,y    circle center.
   *  @param[in]  r      circle radius.
   *  @param[in]  c      RGB color.
   */
  DO_SARA_EXPORT
  auto fill_circle(ImageView<Rgb8>& image, int x, int y, int r, const Rgb8& c)
      -> void;

  //! @}

}}  // namespace DO::Sara
