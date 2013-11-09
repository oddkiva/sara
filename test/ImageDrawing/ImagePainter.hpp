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

#ifndef DO_IMAGEPAINTER_HPP
#define DO_IMAGEPAINTER_HPP

#include <DO/Core.hpp>
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

    void drawLine(double x1, double y1, double x2, double y2,
                  const Vector3d& rgb, double penWidth)
    {
      // Vectorial geometry.
      agg::path_storage ps;
      ps.remove_all();
      ps.move_to(x1, y1);
      ps.line_to(x2, y2);
      // Convert vectorial geometry to pixels.
      agg::conv_stroke<agg::path_storage> pg(ps);
      pg.width(penWidth);
      aa_rasterizer.add_path(pg);
      // Draw.
      agg::render_scanlines_aa_solid(
          aa_rasterizer, scanline, renderer_base,
          agg::rgba(rgb[0], rgb[1], rgb[2]) );
    }
    void drawRectangle(double x1, double y1, double x2, double y2,
                       const Vector3d& rgb, double penWidth)
    {
      // Vectorial geometry.
      agg::path_storage ps;
      ps.remove_all();
      ps.move_to(x1, y1);
      ps.line_to(x2, y1);
      ps.line_to(x2, y2);
      ps.line_to(x1, y2);
      ps.line_to(x1, y1);
      ps.close_polygon();
      // Convert vectorial geometry to pixels.
      agg::conv_stroke<agg::path_storage> pg(ps);
      pg.width(penWidth);
      aa_rasterizer.add_path(pg);
      // Draw.
      agg::render_scanlines_aa_solid(
          aa_rasterizer, scanline, renderer_base,
          agg::rgba(rgb[0], rgb[1], rgb[2]) );
    }
    void drawEllipse(double cx, double cy, double rx, double ry,
                     double oriRadian, const Vector3d& rgb, double penWidth)
    {
      // Create vectorial ellipse.
      agg::ellipse ellipse;
      // Rotate and translate ellipse
      agg::trans_affine transform;
      ellipse.init(0, 0, rx, ry);
      transform.rotate(oriRadian);
      transform.translate(cx, cy);
      // Render ellipse.
      agg::conv_transform<agg::ellipse> ellipticPath(ellipse, transform);
      agg::conv_stroke<agg::conv_transform<agg::ellipse> > ellStroke(ellipticPath);
      ellStroke.width(penWidth);
      aa_rasterizer.add_path(ellStroke);
      agg::render_scanlines_aa_solid(
          aa_rasterizer, scanline, renderer_base,
          agg::rgba(rgb[0], rgb[1], rgb[2]) );
    }
    void drawCircle(double cx, double cy, double r, const Vector3d& rgb,
                    double penWidth)
    { drawEllipse(cx, cy, r, r, 0., rgb, penWidth); }
    void fillRectangle(double x1, double y1, double x2, double y2,
                       const Vector3d& rgb)
    {
      // Vectorial geometry.
      agg::path_storage ps;
      ps.remove_all();
      ps.move_to(x1, y1);
      ps.line_to(x2, y1);
      ps.line_to(x2, y2);
      ps.line_to(x1, y2);
      ps.line_to(x1, y1);
      ps.close_polygon();
      // Convert vectorial geometry to pixels.
      agg::trans_affine identity_transform;
      agg::conv_transform<agg::path_storage> pg(ps, identity_transform);
      aa_rasterizer.add_path(pg);
      // Draw.
      agg::render_scanlines_aa_solid(
          aa_rasterizer, scanline, renderer_base,
          agg::rgba(rgb[0], rgb[1], rgb[2]) );
    }
    void fillEllipse(double cx, double cy, double rx, double ry,
                     double oriRadian, const Vector3d& rgb)
    {
      // Create vectorial ellipse.
      agg::ellipse ellipse(cx, cy, rx, ry);
      // Render ellipse.
      agg::trans_affine transform;
      ellipse.init(0, 0, rx, ry);
      transform.rotate(oriRadian);
      transform.translate(cx, cy);
      agg::conv_transform<agg::ellipse> ellipticPath(ellipse, transform);
      aa_rasterizer.add_path(ellipticPath);
      agg::render_scanlines_aa_solid(
          aa_rasterizer, scanline, renderer_base,
          agg::rgba(rgb[0], rgb[1], rgb[2]) );
    }
    void fillCircle(double cx, double cy, double r, const Vector3d& rgb)
    { fillEllipse(cx, cy, r, r, 0., rgb); }

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

#endif /* DO_IMAGEPAINTER_HPP */