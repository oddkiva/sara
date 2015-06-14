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

#include <DO/Sara/Graphics.hpp>

#include "GraphicsUtilities.hpp"


namespace DO { namespace Sara {

  bool draw_point(int x, int y, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawPoint",
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool draw_point(int x, int y, const Color4ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawPoint",
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2], c[3])));
  }

  bool draw_point(const Point2f& p, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawPoint",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(p.x(), p.y())),
      Q_ARG(const QColor&,
      QColor(c[0], c[1], c[2])));
  }

  bool draw_circle(int xc, int yc, int r, const Color3ub& c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawCircle",
      Qt::QueuedConnection,
      Q_ARG(int, xc), Q_ARG(int, yc),
      Q_ARG(int, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_circle(const Point2f& center, float r, const Color3ub& c,
                   int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawCircle",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(center.x(), center.y())),
      Q_ARG(qreal, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_circle(const Point2d& center, double r, const Color3ub& c,
                   int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawCircle",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(center.x(), center.y())),
      Q_ARG(qreal, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_ellipse(int x, int y, int w, int h, const Color3ub&c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawEllipse",
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_ellipse(const Point2f& center, float r1, float r2, float degree,
                    const Color3ub& c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawEllipse",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(center.x(), center.y())),
      Q_ARG(qreal, qreal(r1)), Q_ARG(qreal, qreal(r2)),
      Q_ARG(qreal, qreal(degree)),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_ellipse(const Point2d& center, double r1, double r2, double degree,
                   const Color3ub& c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawEllipse",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(center.x(), center.y())),
      Q_ARG(qreal, qreal(r1)), Q_ARG(qreal, qreal(r2)),
      Q_ARG(qreal, qreal(degree)),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_line(int x1, int y1, int x2, int y2, const Color3ub& c,
                int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawLine",
      Qt::QueuedConnection,
      Q_ARG(int, x1), Q_ARG(int, y1),
      Q_ARG(int, x2), Q_ARG(int, y2),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_line(const Point2f& p1, const Point2f& p2, const Color3ub& c,
                 int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawLine",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(p1.x(), p1.y())),
      Q_ARG(const QPointF&, QPointF(p2.x(), p2.y())),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_line(const Point2d& p1, const Point2d& p2, const Color3ub& c,
                int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawLine",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(p1.x(), p1.y())),
      Q_ARG(const QPointF&, QPointF(p2.x(), p2.y())),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool draw_rect(int x, int y, int w, int h, const Color3ub& c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawRect",
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  static bool draw_poly(const QPolygonF& poly, const Color3ub& c, int width)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawPoly",
      Qt::QueuedConnection,
      Q_ARG(const QPolygonF&, poly),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, width));
  }

  bool draw_poly(const int x[], const int y[], int n, const Color3ub& c,
                 int width)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(x[i]), qreal(y[i]));
    return draw_poly(poly, c, width);
  }

  bool draw_poly(const Point2i* p, int n, const Color3ub& c, int width)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(p[i].x()), qreal(p[i].y()));
    return draw_poly(poly, c, width);
  }

  bool draw_string(int x, int y, const std::string &s, const Color3ub& c,
                   int fontSize, double alpha, bool italic, bool bold,
                   bool underlined)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawText",
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(const QString&, QString(s.c_str())),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, fontSize),
      Q_ARG(qreal, qreal(alpha)),
      Q_ARG(bool, italic), Q_ARG(bool, bold),
      Q_ARG(bool, underlined));
  }

  bool draw_arrow(int a, int b, int c, int d, const Color3ub& col,
                  int arrowWidth, int arrowHeight, int style, int width)
  {
    return QMetaObject::invokeMethod(
      active_window(), "drawArrow",
      Qt::QueuedConnection,
      Q_ARG(int, a), Q_ARG(int, b),
      Q_ARG(int, c), Q_ARG(int, d),
      Q_ARG(const QColor&, QColor(col[0], col[1], col[2])),
      Q_ARG(int, arrowWidth),
      Q_ARG(int, arrowHeight),
      Q_ARG(int, style), Q_ARG(int, width));
  }

  bool fill_ellipse(int x, int y, int w, int h, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "fillEllipse",
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fill_ellipse(const Point2f& p, float rx, float ry, float degree,
                   const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "fillEllipse",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(p.x(), p.y())),
      Q_ARG(qreal, rx),
      Q_ARG(qreal, ry),
      Q_ARG(qreal, degree),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fill_rect(int x, int y, int w, int h, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "fillRect",
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fill_circle(int x, int y, int r, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "fillCircle",
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fill_circle(const Point2f& p, float r, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "fillCircle",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPoint(p.x(), p.y())),
      Q_ARG(qreal, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  static bool fill_poly(const QPolygonF& polygon, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      active_window(), "fillPoly",
      Qt::QueuedConnection,
      Q_ARG(const QPolygonF&, polygon),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fill_poly(const int x[], const int y[], int n, const Color3ub& c)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(x[i]), qreal(y[i]));
    return fill_poly(poly, c);
  }

  bool fill_poly(const int p[], int n, const Color3ub& c)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(p[2*i]), qreal(p[2*i+1]));
    return fill_poly(poly, c);
  }

  bool fill_poly(const Point2i *p, int n, const Color3ub& c)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(p[i].x()), qreal(p[i].y()));
    return fill_poly(poly, c);
  }

  static bool display(const QImage& image, int xoff, int yoff, double fact)
  {
    return QMetaObject::invokeMethod(
      active_window(), "display",
      Qt::BlockingQueuedConnection,
      Q_ARG(const QImage&, image),
      Q_ARG(int, xoff), Q_ARG(int, yoff),
      Q_ARG(double, fact));
  }

  bool put_color_image(int x, int y, const Color3ub *data, int w, int h,
                     double fact)
  {
    QImage image(reinterpret_cast<const uchar*>(data),
                 w, h, w*3, QImage::Format_RGB888);
    return display(image, x, y, fact);
  }

  bool put_grey_image(int x, int y, const uchar *data, int w, int h,
                    double fact)
  {
    QImage image(data, w, h, w, QImage::Format_Indexed8);
    QVector<QRgb> colorTable(256);
    for (int i = 0; i < 256; ++i)
      colorTable[i] = qRgb(i, i, i);
    image.setColorTable(colorTable);
    return display(image, x, y, fact);
  }

  bool clear_window()
  {
    return QMetaObject::invokeMethod(
      active_window(), "clear",
      Qt::QueuedConnection);
  }

  bool set_antialiasing(Window w, bool on)
  {
    return QMetaObject::invokeMethod(
      w, "setAntialiasing",
      Qt::QueuedConnection,
      Q_ARG(bool, on));
  }

  bool set_transparency(Window w, bool on)
  {
    return QMetaObject::invokeMethod(
      w, "setTransparency",
      Qt::QueuedConnection,
      Q_ARG(bool, on));
  }

  bool save_screen(Window w, const std::string& fileName)
  {
    if (!active_window_is_visible())
      return false;
    return QMetaObject::invokeMethod(
      w, "saveScreen",
      Qt::BlockingQueuedConnection,
      Q_ARG(const QString&, QString(fileName.c_str())));
    return true;
  }

} /* namespace Sara */
} /* namespace DO */
