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
    draw_line(image, a.x(), a.y(), b.x(), b.y(), c, pen_width, antialiasing);
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

  //! @brief Draw ellipse.
  auto draw_ellipse(ImageView<Rgb8>& image, const Eigen::Vector2f& center,
                    float r1, float r2, float degree, const Rgb8& c,
                    int pen_width = 1, bool antialiasing = true) -> void;

  //! @brief Draw arrow.
  DO_SARA_EXPORT
  auto draw_arrow(ImageView<Rgb8>& image, int x1, int y1, int x2, int y2,
                  const Rgb8& col, int arrow_width = 8, int arrow_height = 5,
                  int style = 0, int width = 1, bool antialiasing = true)
      -> void;

  inline auto draw_arrow(ImageView<Rgb8>& image, const Eigen::Vector2f& a,
                         const Eigen::Vector2f& b, const Rgb8& col,
                         int arrow_width = 8, int arrow_height = 5,
                         int style = 0, int width = 1, bool antialiasing = true)
      -> void
  {
    draw_arrow(image, a.x(), a.y(), b.x(), b.y(), col, arrow_width,
               arrow_height, style, width, antialiasing);
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
