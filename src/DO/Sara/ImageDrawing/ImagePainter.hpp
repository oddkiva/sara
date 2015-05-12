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

#ifndef DO_IMAGEDRAWING_IMAGEPAINTER_HPP
#define DO_IMAGEDRAWING_IMAGEPAINTER_HPP

// Pixel format
#include "agg_pixfmt_rgb.h"
// Rendering-related headers.
#include "agg_renderer_scanline.h"
#include "agg_renderer_base.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_scanline_p.h"
// Geometric transforms.
#include "agg_conv_transform.h"
#include "agg_trans_affine.h"
// Ellipse
#include "agg_ellipse.h"
#include "agg_path_storage.h"
#include "agg_conv_stroke.h"

namespace DO {

  struct ImagePainter
  {
    ImagePainter(unsigned char *rgb_raw_data, int width, int height)
      : framebuffer(&rgb_raw_data[0], width, height, width*3)
      , pixel_format(framebuffer)
      , renderer_base(pixel_format) {}

    void drawLine(double x1, double y1, double x2, double y2, double penWidth,
                  const Vector3d& rgb, double alpha = 1);
    void drawRectangle(double x1, double y1, double x2, double y2,
                       double penWidth, const Vector3d& rgb, double alpha);
    void drawEllipse(double cx, double cy, double rx, double ry,
                     double penWidth, double oriRadian, const Vector3d& rgb, 
                     double alpha = 1);
    void drawCircle(double cx, double cy, double r, double penWidth,
                    const Vector3d& rgb, double alpha = 1)
    { drawEllipse(cx, cy, r, r, 0., penWidth, rgb, alpha); }
    void fillRectangle(double x1, double y1, double x2, double y2,
                       const Vector3d& rgb, double alpha = 1);
    void fillEllipse(double cx, double cy, double rx, double ry,
                     double oriRadian, const Vector3d& rgb, double alpha = 1);
    void fillCircle(double cx, double cy, double r, const Vector3d& rgb,
                    double alpha = 1)
    { fillEllipse(cx, cy, r, r, 0., rgb, alpha); }

    // Some convenient typedefs.
    typedef agg::renderer_base<agg::pixfmt_rgb24> renderer_base_type;
    typedef agg::renderer_scanline_aa_solid<renderer_base_type> 
      renderer_scanline_type;
    typedef agg::rasterizer_scanline_aa<> rasterizer_scanline_type;
    // Rendering buffer data structure to which raw image data is attached.
    agg::rendering_buffer framebuffer;
    // Pixel format of the frame buffer.
    agg::pixfmt_rgb24 pixel_format;
    // Renderer supporting low-level clipping.
    agg::renderer_base<agg::pixfmt_rgb24> renderer_base;
    // Anti-aliased rasterizer.
    agg::rasterizer_scanline_aa<> aa_rasterizer;
    // Use a packed scanline storage.
    agg::scanline_p8 scanline;
  };

} /* namespace DO */

#endif /* DO_IMAGEDRAWING_IMAGEPAINTER_HPP */