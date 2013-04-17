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

namespace DO {

	// ====================================================================== //
	//! Drawing commands
	void drawPoint(int x, int y, const Color3ub& c)
	{
		QMetaObject::invokeMethod(activeWindow(), "drawPoint", 
								              Qt::QueuedConnection,
								              Q_ARG(int, x), Q_ARG(int, y),
								              Q_ARG(const QColor&, QColor(c[0], c[1], c[2])));
	}

	void drawPoint(int x, int y, const Color4ub& c)
	{
		QMetaObject::invokeMethod(activeWindow(), "drawPoint", 
								              Qt::QueuedConnection,
								              Q_ARG(int, x), Q_ARG(int, y),
								              Q_ARG(const QColor&, 
                                    QColor(c[0], c[1], c[2], c[3])));
	}

  void drawPoint(const Point2f& p, const Color3ub& c)
  {
		QMetaObject::invokeMethod(activeWindow(), "drawPoint", 
								              Qt::QueuedConnection,
								              Q_ARG(const QPointF&, QPointF(p.x(), p.y())),
								              Q_ARG(const QColor&, 
                                    QColor(c[0], c[1], c[2])));
  }

	void drawCircle(int xc, int yc, int r, const Color3ub& c, int penWidth)
	{
		QMetaObject::invokeMethod(activeWindow(), "drawCircle", 
								              Qt::QueuedConnection,
								              Q_ARG(int, xc), Q_ARG(int, yc),
								              Q_ARG(int, r),
								              Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
								              Q_ARG(int, penWidth));
	}

	void drawEllipse(int x, int y, int w, int h, const Color3ub&c, int penWidth)
	{
		QMetaObject::invokeMethod(activeWindow(), "drawEllipse", 
								              Qt::QueuedConnection,
								              Q_ARG(int, x), Q_ARG(int, y),
								              Q_ARG(int, w), Q_ARG(int, h),
								              Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
								              Q_ARG(int, penWidth));
	}

  void drawEllipse(const Point2f& center, float r1, float r2, float degree,
                   const Color3ub& c, int penWidth)
  {
		QMetaObject::invokeMethod(activeWindow(), "drawEllipse", 
								              Qt::QueuedConnection,
								              Q_ARG(const QPointF&,
                                    QPointF(center.x(), center.y())),
								              Q_ARG(qreal, qreal(r1)), Q_ARG(qreal, qreal(r2)),
                              Q_ARG(qreal, qreal(degree)),
								              Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
								              Q_ARG(int, penWidth));
  }

	void drawLine(int x1, int y1, int x2, int y2, const Color3ub& c,
				        int penWidth)
	{
		QMetaObject::invokeMethod(activeWindow(), "drawLine", 
								              Qt::QueuedConnection,
								              Q_ARG(int, x1), Q_ARG(int, y1),
								              Q_ARG(int, x2), Q_ARG(int, y2),
								              Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
								              Q_ARG(int, penWidth));
	}

  void drawLine(const Point2f& p1, const Point2f& p2, const Color3ub& c,
                int penWidth)
  {
		QMetaObject::invokeMethod(activeWindow(), "drawLine", 
								              Qt::QueuedConnection,
								              Q_ARG(const QPointF&, QPointF(p1.x(), p1.y())),
                              Q_ARG(const QPointF&, QPointF(p2.x(), p2.y())),
								              Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
								              Q_ARG(int, penWidth));
  }

  void drawLine(const Point2d& p1, const Point2d& p2, const Color3ub& c,
                int penWidth)
  {
    QMetaObject::invokeMethod(activeWindow(), "drawLine", 
                              Qt::QueuedConnection,
                              Q_ARG(const QPointF&, QPointF(p1.x(), p1.y())),
                              Q_ARG(const QPointF&, QPointF(p2.x(), p2.y())),
                              Q_ARG(const QColor&, QColor(c[0], c[1], c[2])),
                              Q_ARG(int, penWidth));
  }

	void drawRect(int x, int y, int w, int h, const Color3ub& c, int penWidth)
	{
		QMetaObject::invokeMethod(activeWindow(), "drawRect", 
                              Qt::QueuedConnection,
                              Q_ARG(int, x), Q_ARG(int, y),
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QColor&, 
                                    QColor(c[0], c[1], c[2])),
                              Q_ARG(int, penWidth));
	}

	void drawPoly(const QPolygonF& poly, const Color3ub& c, int width)
	{
		QMetaObject::invokeMethod(activeWindow(), "drawPoly", 
                              Qt::QueuedConnection,
                              Q_ARG(const QPolygonF&, poly),
                              Q_ARG(const QColor&, 
                                    QColor(c[0], c[1], c[2])),
                              Q_ARG(int, width));
	}

	void drawPoly(const int x[], const int y[], int n, const Color3ub& c,
				        int width)
	{
		QPolygonF poly;
		for (int i = 0; i < n; ++i)
			poly << QPointF(qreal(x[i]), qreal(y[i]));
		drawPoly(poly, c, width);
	}

	void drawPoly(const Point2i* p, int n, const Color3ub& c, int width)
	{
		QPolygonF poly;
		for (int i = 0; i < n; ++i)
			poly << QPointF(qreal(p[i].x()), qreal(p[i].y()));
		drawPoly(poly, c, width);
	}

	void drawString(int x, int y, const std::string &s, const Color3ub& c,
					        int fontSize, double alpha, bool italic, bool bold, 
					        bool underlined)
	{
    QMetaObject::invokeMethod(activeWindow(), "drawText", 
                              Qt::QueuedConnection,
                              Q_ARG(int, x), Q_ARG(int, y),
                              Q_ARG(const QString&, 
                                    QString(s.c_str())),
                              Q_ARG(const QColor&, 
                                    QColor(c[0], c[1], c[2])),
                              Q_ARG(int, fontSize), 
                              Q_ARG(qreal, qreal(alpha)),
                              Q_ARG(bool, italic), Q_ARG(bool, bold),
                              Q_ARG(bool, underlined));
	}

	void drawArrow(int a, int b, int c, int d, const Color3ub& col,
				         int arrowWidth, int arrowHeight, int style, int width)
	{
		QMetaObject::invokeMethod(activeWindow(), "drawArrow", 
                              Qt::QueuedConnection,
                              Q_ARG(int, a), Q_ARG(int, b),
                              Q_ARG(int, c), Q_ARG(int, d),
                              Q_ARG(const QColor&,
                                    QColor(col[0], col[1], col[2])),
                              Q_ARG(int, arrowWidth), 
                              Q_ARG(int, arrowHeight),
                              Q_ARG(int, style), Q_ARG(int, width));
	}

	// ====================================================================== //
	//! Filling commands
	void fillEllipse(int x, int y, int w, int h, const Color3ub& c)
	{
		QMetaObject::invokeMethod(activeWindow(), "fillEllipse", 
                              Qt::QueuedConnection,
                              Q_ARG(int, x), Q_ARG(int, y),
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QColor&,
                                    QColor(c[0], c[1], c[2])));
	}

  void fillEllipse(const Point2f& p, float rx, float ry, float degree,
                   const Color3ub& c)
  {
    QMetaObject::invokeMethod(activeWindow(), "fillEllipse",
                              Qt::QueuedConnection,
                              Q_ARG(const QPointF&, QPointF(p.x(), p.y())),
                              Q_ARG(qreal, rx),
                              Q_ARG(qreal, ry),
                              Q_ARG(qreal, degree),
                              Q_ARG(const QColor&, QColor(c[0], c[1], c[2]))
                              );
  }

	void fillRect(int x, int y, int w, int h, const Color3ub& c)
	{
		QMetaObject::invokeMethod(activeWindow(), "fillRect", 
                              Qt::QueuedConnection,
                              Q_ARG(int, x), Q_ARG(int, y),
                              Q_ARG(int, w), Q_ARG(int, h),
                              Q_ARG(const QColor&,
                                    QColor(c[0], c[1], c[2])));
	}

	void fillCircle(int x, int y, int r, const Color3ub& c)
	{
		QMetaObject::invokeMethod(activeWindow(), "fillCircle", 
								              Qt::QueuedConnection,
								              Q_ARG(int, x), Q_ARG(int, y),
								              Q_ARG(int, r),
								              Q_ARG(const QColor&,
										                QColor(c[0], c[1], c[2])));
	}

  void fillCircle(const Point2f& p, float r, const Color3ub& c)
  {
    QMetaObject::invokeMethod(activeWindow(), "fillCircle", 
                              Qt::QueuedConnection,
                              Q_ARG(const QPointF&, QPoint(p.x(), p.y())),
                              Q_ARG(qreal, r),
                              Q_ARG(const QColor&, QColor(c[0], c[1], c[2]))
                              );
  }

	void fillPoly(const QPolygonF& polygon, const Color3ub& c)
	{
		QMetaObject::invokeMethod(activeWindow(), "fillPoly", 
                              Qt::QueuedConnection,
                              Q_ARG(const QPolygonF&, polygon),
                              Q_ARG(const QColor&,
                                    QColor(c[0], c[1], c[2])));
	}

	void fillPoly(const int x[], const int y[], int n, const Color3ub& c)
	{
		QPolygonF poly;
		for (int i = 0; i < n; ++i)
			poly << QPointF(qreal(x[i]), qreal(y[i]));
		fillPoly(poly, c);
	}

	void fillPoly(const int p[], int n, const Color3ub& c)
	{
		QPolygonF poly;
		for (int i = 0; i < n; ++i)
			poly << QPointF(qreal(p[2*i]), qreal(p[2*i+1]));
		fillPoly(poly, c);
	}

	void fillPoly(const Point2i *p, int n, const Color3ub& c)
	{
		QPolygonF poly;
		for (int i = 0; i < n; ++i)
			poly << QPointF(qreal(p[i].x()), qreal(p[i].y()));
		fillPoly(poly, c);
	}

	// ====================================================================== //
	//! Image display commands
	void display(const QImage& image, int xoff, int yoff, double fact)
	{
		QMetaObject::invokeMethod(activeWindow(), "display", 
                              Qt::BlockingQueuedConnection,
                              Q_ARG(const QImage&, image),
                              Q_ARG(int, xoff), Q_ARG(int, yoff),
                              Q_ARG(double, fact));
	}

	void putColorImage(int x, int y, const Color3ub *data, int w, int h,
					           double fact)
	{
		QImage image(reinterpret_cast<const uchar*>(data),
                 w, h, w*3, QImage::Format_RGB888);
		display(image, x, y, fact);
	}

	void putGreyImage(int x, int y, const uchar *data, int w, int h,
					          double fact)
	{
		QImage image(data, w, h, w, QImage::Format_Indexed8);
		QVector<QRgb> colorTable(256);
		for (int i = 0; i < 256; ++i)
			colorTable[i] = qRgb(i, i, i);
		image.setColorTable(colorTable);
		display(image, x, y, fact);
	}

	// ====================================================================== //
	//! Clearing commands
	void clearWindow()
	{
		QMetaObject::invokeMethod(activeWindow(), "clear", Qt::QueuedConnection);
	}

  // ======================================================================== //
  //! Painting options commands
  void setAntialiasing(Window w, bool on)
  {
    QMetaObject::invokeMethod(w, "setAntialiasing",
                              Qt::QueuedConnection,
                              Q_ARG(bool, on));
  }
  
  void setTransparency(Window w, bool on)
  {
    QMetaObject::invokeMethod(w, "setTransparency",
                              Qt::QueuedConnection,
                              Q_ARG(bool, on));
  }

  bool saveScreen(Window w, const std::string& fileName)
  {
    if (!activeWindowIsVisible())
      return false;
    QMetaObject::invokeMethod(w, "saveScreen",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(const QString&, QString(fileName.c_str())));
    return true;
  }

} /* namespace DO */