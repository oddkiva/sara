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
#include "GraphicsUtilities.hpp"

namespace DO {

  // ====================================================================== //
  //! Drawing commands
  bool drawPoint(int x, int y, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawPoint", 
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool drawPoint(int x, int y, const Color4ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawPoint", 
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2], c[3])));
  }

  bool drawPoint(const Point2f& p, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawPoint", 
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(p.x(), p.y())),
      Q_ARG(const QColor&, 
      QColor(c[0], c[1], c[2])));
  }

  bool drawCircle(int xc, int yc, int r, const Color3ub& c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawCircle", 
      Qt::QueuedConnection,
      Q_ARG(int, xc), Q_ARG(int, yc),
      Q_ARG(int, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool drawCircle(const Point2f& center, float r, const Color3ub& c,
                  int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawCircle", 
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(center.x(), center.y())),
      Q_ARG(qreal, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }
  
  bool drawCircle(const Point2d& center, double r, const Color3ub& c,
                  int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawCircle", 
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(center.x(), center.y())),
      Q_ARG(qreal, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool drawEllipse(int x, int y, int w, int h, const Color3ub&c, int penWidth)
  {
    return QMetaObject::invokeMethod(getActiveWindow(), "drawEllipse", 
                              Qt::QueuedConnection,
                              Q_ARG(int, x), Q_ARG(int, y),
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
                              Q_ARG(int, penWidth));
  }

  bool drawEllipse(const Point2f& center, float r1, float r2, float degree,
                   const Color3ub& c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawEllipse", 
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(center.x(), center.y())),
      Q_ARG(qreal, qreal(r1)), Q_ARG(qreal, qreal(r2)),
      Q_ARG(qreal, qreal(degree)),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool drawEllipse(const Point2d& center, double r1, double r2, double degree,
                   const Color3ub& c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawEllipse", 
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(center.x(), center.y())),
      Q_ARG(qreal, qreal(r1)), Q_ARG(qreal, qreal(r2)),
      Q_ARG(qreal, qreal(degree)),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool drawLine(int x1, int y1, int x2, int y2, const Color3ub& c,
                int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawLine", 
      Qt::QueuedConnection,
      Q_ARG(int, x1), Q_ARG(int, y1),
      Q_ARG(int, x2), Q_ARG(int, y2),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool drawLine(const Point2f& p1, const Point2f& p2, const Color3ub& c,
                int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawLine", 
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(p1.x(), p1.y())),
      Q_ARG(const QPointF&, QPointF(p2.x(), p2.y())),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool drawLine(const Point2d& p1, const Point2d& p2, const Color3ub& c,
                int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawLine", 
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(p1.x(), p1.y())),
      Q_ARG(const QPointF&, QPointF(p2.x(), p2.y())),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  bool drawRect(int x, int y, int w, int h, const Color3ub& c, int penWidth)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawRect", 
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, penWidth));
  }

  static bool drawPoly(const QPolygonF& poly, const Color3ub& c, int width)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawPoly", 
      Qt::QueuedConnection,
      Q_ARG(const QPolygonF&, poly),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, width));
  }

  bool drawPoly(const int x[], const int y[], int n, const Color3ub& c,
                int width)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(x[i]), qreal(y[i]));
    return drawPoly(poly, c, width);
  }

  bool drawPoly(const Point2i* p, int n, const Color3ub& c, int width)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(p[i].x()), qreal(p[i].y()));
    return drawPoly(poly, c, width);
  }

  bool drawString(int x, int y, const std::string &s, const Color3ub& c,
                  int fontSize, double alpha, bool italic, bool bold, 
                  bool underlined)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawText", 
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(const QString&, QString(s.c_str())),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
      Q_ARG(int, fontSize), 
      Q_ARG(qreal, qreal(alpha)),
      Q_ARG(bool, italic), Q_ARG(bool, bold),
      Q_ARG(bool, underlined));
  }

  bool drawArrow(int a, int b, int c, int d, const Color3ub& col,
                 int arrowWidth, int arrowHeight, int style, int width)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "drawArrow", 
      Qt::QueuedConnection,
      Q_ARG(int, a), Q_ARG(int, b),
      Q_ARG(int, c), Q_ARG(int, d),
      Q_ARG(const QColor&, QColor(col[0], col[1], col[2])),
      Q_ARG(int, arrowWidth), 
      Q_ARG(int, arrowHeight),
      Q_ARG(int, style), Q_ARG(int, width));
  }

  // ====================================================================== //
  //! Filling commands
  bool fillEllipse(int x, int y, int w, int h, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "fillEllipse", 
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fillEllipse(const Point2f& p, float rx, float ry, float degree,
                   const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "fillEllipse",
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPointF(p.x(), p.y())),
      Q_ARG(qreal, rx),
      Q_ARG(qreal, ry),
      Q_ARG(qreal, degree),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fillRect(int x, int y, int w, int h, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "fillRect", 
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, w), Q_ARG(int, h),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fillCircle(int x, int y, int r, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "fillCircle", 
      Qt::QueuedConnection,
      Q_ARG(int, x), Q_ARG(int, y),
      Q_ARG(int, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fillCircle(const Point2f& p, float r, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "fillCircle", 
      Qt::QueuedConnection,
      Q_ARG(const QPointF&, QPoint(p.x(), p.y())),
      Q_ARG(qreal, r),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  static bool fillPoly(const QPolygonF& polygon, const Color3ub& c)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "fillPoly", 
      Qt::QueuedConnection,
      Q_ARG(const QPolygonF&, polygon),
      Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
  }

  bool fillPoly(const int x[], const int y[], int n, const Color3ub& c)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(x[i]), qreal(y[i]));
    return fillPoly(poly, c);
  }

  bool fillPoly(const int p[], int n, const Color3ub& c)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(p[2*i]), qreal(p[2*i+1]));
    return fillPoly(poly, c);
  }

  bool fillPoly(const Point2i *p, int n, const Color3ub& c)
  {
    QPolygonF poly;
    for (int i = 0; i < n; ++i)
      poly << QPointF(qreal(p[i].x()), qreal(p[i].y()));
    return fillPoly(poly, c);
  }

  // ====================================================================== //
  //! Image display commands
  static bool display(const QImage& image, int xoff, int yoff, double fact)
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "display", 
      Qt::BlockingQueuedConnection,
      Q_ARG(const QImage&, image),
      Q_ARG(int, xoff), Q_ARG(int, yoff),
      Q_ARG(double, fact));
  }

  bool putColorImage(int x, int y, const Color3ub *data, int w, int h,
                     double fact)
  {
    QImage image(reinterpret_cast<const uchar*>(data),
                 w, h, w*3, QImage::Format_RGB888);
    return display(image, x, y, fact);
  }

  bool putGreyImage(int x, int y, const uchar *data, int w, int h,
                    double fact)
  {
    QImage image(data, w, h, w, QImage::Format_Indexed8);
    QVector<QRgb> colorTable(256);
    for (int i = 0; i < 256; ++i)
      colorTable[i] = qRgb(i, i, i);
    image.setColorTable(colorTable);
    return display(image, x, y, fact);
  }

  // ====================================================================== //
  //! Clearing commands
  bool clearWindow()
  {
    return QMetaObject::invokeMethod(
      getActiveWindow(), "clear",
      Qt::QueuedConnection);
  }

  // ======================================================================== //
  //! Painting options commands
  bool setAntialiasing(Window w, bool on)
  {
    return QMetaObject::invokeMethod(
      w, "setAntialiasing",
      Qt::QueuedConnection,
      Q_ARG(bool, on));
  }
  
  bool setTransparency(Window w, bool on)
  {
    return QMetaObject::invokeMethod(
      w, "setTransparency",
      Qt::QueuedConnection,
      Q_ARG(bool, on));
  }

  bool saveScreen(Window w, const std::string& fileName)
  {
    if (!activeWindowIsVisible())
      return false;
    return QMetaObject::invokeMethod(
      w, "saveScreen",
      Qt::BlockingQueuedConnection,
      Q_ARG(const QString&, QString(fileName.c_str())));
    return true;
  }

} /* namespace DO */