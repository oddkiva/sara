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

#include <QtGui>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>


namespace DO { namespace Sara {

  void draw_point(ImageView<Rgb8>& image, int x, int y, const Color3ub& c)
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.setPen(to_QColor(c));
    p.drawPoint(x, y);
  }

  void draw_circle(ImageView<Rgb8>& image,
                   int xc, int yc, int r, const Color3ub& c, int penWidth)
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.setPen(QPen(to_QColor(c), penWidth));
    p.drawEllipse(QPoint(xc, yc), r, r);
  }

  void draw_line(ImageView<Rgb8>& image,
                 int x1, int y1, int x2, int y2, const Color3ub& c,
                 int penWidth)
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.setPen(QPen(to_QColor(c), penWidth));
    p.drawLine(x1, y1, x2, y2);
  }

  void draw_rect(ImageView<Rgb8>& image,
                 int x, int y, int w, int h, const Color3ub& c,
                 int penWidth)
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.setPen(QPen(to_QColor(c), penWidth));
    p.drawRect(x, y, w, h);
  }

  void fill_rect(ImageView<Rgb8>& image,
                 int x, int y, int w, int h, const Color3ub& c)
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.fillRect(x, y, w, h, to_QColor(c));
  }

  void fill_circle(ImageView<Rgb8>& image,
                   int x, int y, int r, const Color3ub& c)
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    QPainterPath path;
    path.addEllipse(QPointF(x,y), qreal(r), qreal(r));
    p.fillPath(path, to_QColor(c));
  }

} /* namespace Sara */
} /* namespace DO */
