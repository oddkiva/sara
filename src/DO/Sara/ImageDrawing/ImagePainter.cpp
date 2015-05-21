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

#include <DO/Sara/ImageDrawing.hpp>

namespace DO {

  void ImagePainter::drawLine(double x1, double y1, double x2, double y2,
                              double penWidth, const Vector3d& rgb,
                              double alpha)
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
      agg::rgba(rgb[0], rgb[1], rgb[2], alpha) );
  }

  void ImagePainter::drawRectangle(double x1, double y1, double x2, double y2,
                                   double penWidth, const Vector3d& rgb,
                                   double alpha)
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
      agg::rgba(rgb[0], rgb[1], rgb[2], alpha) );
  }

  void ImagePainter::drawEllipse(double cx, double cy, double rx, double ry,
                                 double oriRadian, double penWidth,
                                 const Vector3d& rgb, double alpha)
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
      agg::rgba(rgb[0], rgb[1], rgb[2], alpha) );
  }

  void ImagePainter::fillRectangle(double x1, double y1, double x2, double y2,
                                   const Vector3d& rgb, double alpha)
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
      agg::rgba(rgb[0], rgb[1], rgb[2], alpha) );
  }

  void ImagePainter::fillEllipse(double cx, double cy, double rx, double ry,
                                 double oriRadian, const Vector3d& rgb,
                                 double alpha)
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
      agg::rgba(rgb[0], rgb[1], rgb[2], alpha) );
  }


} /* namespace DO */