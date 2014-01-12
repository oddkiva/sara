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

#include <DO/Graphics.hpp>
#include <QtGui>

namespace DO {

  static inline QImage toQImage(Image<Rgb8>& image)
  {
    return QImage(reinterpret_cast<unsigned char *>(image.data()),
                  image.width(), image.height(), image.width()*3,
                  QImage::Format_RGB888);
  }

  static inline QColor toQColor(const Color3ub& c)
  {
    return QColor(c[0], c[1], c[2]);
  }

  void drawPoint(Image<Rgb8>& image, int x, int y, const Color3ub& c)
  {
    QImage qimage(toQImage(image));
    QPainter p(&qimage);
    p.setPen(toQColor(c));
    p.drawPoint(x, y);
  }

  void drawCircle(Image<Rgb8>& image,
                  int xc, int yc, int r, const Color3ub& c, int penWidth)
  {
    QImage qimage(toQImage(image));
    QPainter p(&qimage);
    p.setPen(QPen(toQColor(c), penWidth));
    p.drawEllipse(QPoint(xc, yc), r, r);
  }

  void drawLine(Image<Rgb8>& image,
                int x1, int y1, int x2, int y2, const Color3ub& c,
                int penWidth)
  {
    QImage qimage(toQImage(image));
    QPainter p(&qimage);
    p.setPen(QPen(toQColor(c), penWidth));
    p.drawLine(x1, y1, x2, y2);
  }

  void drawRect(Image<Rgb8>& image,
                int x, int y, int w, int h, const Color3ub& c,
                int penWidth)
  {
    QImage qimage(toQImage(image));
    QPainter p(&qimage);
    p.setPen(QPen(toQColor(c), penWidth));
    p.drawRect(x, y, w, h);
  }

  void fillRect(Image<Rgb8>& image,
                int x, int y, int w, int h, const Color3ub& c)
  {
    QImage qimage(toQImage(image));
    QPainter p(&qimage);
    p.fillRect(x, y, w, h, toQColor(c));
  }

  void fillCircle(Image<Rgb8>& image,
                  int x, int y, int r, const Color3ub& c)
  {
    QImage qimage(toQImage(image));
    QPainter p(&qimage);
    QPainterPath path;
    path.addEllipse(QPointF(x,y), qreal(r), qreal(r));
    p.fillPath(path, toQColor(c));
  }

} /* namespace DO */