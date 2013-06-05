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

#include "PaintingWindow.hpp"
#include <QtWidgets>

namespace DO {

	// ====================================================================== //
	// ScrollArea
	ScrollArea::ScrollArea(QWidget* parent) : QScrollArea(parent)
	{
    setAlignment(Qt::AlignCenter);
	}

	void ScrollArea::closeEvent(QCloseEvent *event)
	{
    if(event->spontaneous())
    {
      qWarning() << "\n\nWarning: you closed a window unexpectedly!\n\n";
      qWarning() << "Graphical application is terminating...";
      qApp->exit(0);
    }
	}

	// ====================================================================== //
	// PaintingWindow
  PaintingWindow::PaintingWindow(int width, int height,
                                 const QString& windowTitle, int x, int y,
                                 QWidget* parent)
    : QWidget(parent)
    , scroll_area_(new ScrollArea(parent))
    , pixmap_(width, height)
    , painter_(&pixmap_)
  {
    resize(width, height);
    //setMouseTracking(true);
    setFocusPolicy(Qt::WheelFocus);

    event_listening_timer_.setSingleShot(true);
    connect(&event_listening_timer_, SIGNAL(timeout()),
      this, SLOT(eventListeningTimerStopped()));

    if(x != -1 && y != -1)
      scroll_area_->move(x,y);
    scroll_area_->setWindowTitle(windowTitle);
    scroll_area_->setWidget(this);
    scroll_area_->setFocusProxy(this);

    if (width > qApp->desktop()->width() || height > qApp->desktop()->height())
    {
      width = 800;
      height = 600;
    }

    scroll_area_->resize(width+2, height+2);

    pixmap_.fill();
    update();

    scroll_area_->show();
  }

	void PaintingWindow::drawPoint(int x, int y, const QColor& c)
	{
		painter_.setPen(c);
		painter_.drawPoint(x, y);
		update();
	}

  void PaintingWindow::drawPoint(const QPointF& p, const QColor& c)
  {
    painter_.setPen(c);
    painter_.drawPoint(p);
    update();
  }

	void PaintingWindow::drawLine(int x1, int y1, int x2, int y2,
								                const QColor& c, int penWidth)
	{
		painter_.setPen(QPen(c, penWidth));
		painter_.drawLine(x1, y1, x2, y2);
		update();
	}

  void PaintingWindow::drawLine(const QPointF& p1, const QPointF& p2, 
                                const QColor& c, int penWidth)
  {
    painter_.setPen(QPen(c, penWidth));
    painter_.drawLine(p1, p2);
    update();
  }

	void PaintingWindow::drawCircle(int xc, int yc, int r, const QColor& c,
                                  int penWidth)
  {
    painter_.setPen(QPen(c, penWidth));
    painter_.drawEllipse(QPoint(xc, yc), r, r);
    update();
  }

  void PaintingWindow::drawCircle(const QPointF& center, float r,
                                  const QColor& c, int penWidth)
  {
    painter_.setPen(QPen(c, penWidth));
    painter_.drawEllipse(QPointF(center.x(), center.y()), r, r);
    update();
  }

	void PaintingWindow::drawEllipse(int x, int y, int w, int h,
									                 const QColor& c, int penWidth)
	{
		painter_.setPen(QPen(c, penWidth));
		painter_.drawEllipse(x, y, w, h);
		update();
	}

  void PaintingWindow::drawEllipse(const QPointF& center, qreal r1, qreal r2,
                                   qreal degree, const QColor& c, int penWidth)
  {
    painter_.save();
    painter_.setPen(QPen(c, penWidth));
    painter_.translate(center);
    painter_.rotate(degree);
    painter_.translate(-r1, -r2);
    painter_.drawEllipse(QRectF(0, 0, 2*r1, 2*r2));
    painter_.restore();
    update();
  }

	void PaintingWindow::drawPoly(const QPolygonF& polygon, const QColor& c,
								                int width)
	{
		painter_.setPen(QPen(c, width));
		painter_.drawPolygon(polygon);
		update();
	}

	void PaintingWindow::drawRect(int x, int y, int w, int h, const QColor& c,
								                int penWidth)
	{
		painter_.setPen(QPen(c, penWidth));
		painter_.drawRect(x, y, w, h);
		update();
	}

	void PaintingWindow::drawText(int x,int y,const QString& s,const QColor& c,
								                int fontSize, double alpha, bool italic,
								                bool bold, bool underline)
	{
		QFont font;
		font.setPointSize(fontSize);
		font.setItalic(italic);
		font.setBold(bold);
		font.setUnderline(underline);
		painter_.setPen(c);
		painter_.setFont(font);
		painter_.rotate(qreal(alpha));
		painter_.drawText(x, y, s);
		update();
	}

	void PaintingWindow::drawArrow(int a, int b, int c, int d,
								                 const QColor& col,
								                 int arrowWidth, int arrowHeight, int style,
								                 int width)
	{
		double sl;
		double dx = c-a;
		double dy = d-b;
		double norm= qSqrt(dx*dx+dy*dy);
		if (norm < 0.999) // null vector
		{
			painter_.setPen(QPen(col, width));
			painter_.drawPoint(a, b);
			update();
			return;
		}

		QPainterPath path;
		QPolygonF pts;
		
		qreal dx_norm = dx / norm;
		qreal dy_norm = dy / norm;
		qreal p1x = a+dx_norm*(norm-arrowWidth) + arrowHeight/2.*dy_norm;
		qreal p1y = b+dy_norm*(norm-arrowWidth) - arrowHeight/2.*dx_norm;
		qreal p2x = a+dx_norm*(norm-arrowWidth) - arrowHeight/2.*dy_norm;
		qreal p2y = b+dy_norm*(norm-arrowWidth) + arrowHeight/2.*dx_norm;
		switch(style) {
			case 0:
				painter_.setPen(QPen(col, width));
				painter_.drawLine(a, b, c, d);
				painter_.drawLine(c, d, int(p1x), int(p1y));
				painter_.drawLine(c, d, int(p2x), int(p2y));
				break;
			case 1:
				pts << QPointF(p2x, p2y);
				pts << QPointF(c, d);
				pts << QPointF(p1x, p1y);
				sl = norm-(arrowWidth*.7);
				pts << QPointF(a+dx_norm*sl+dy_norm*width, 
							   b+dy_norm*sl-dx_norm*width);
				pts << QPointF(a+dy_norm*width, b-dx_norm*width);
				pts << QPointF(a-dy_norm*width, b+dx_norm*width);
				pts << QPointF(a+dx_norm*sl-dy_norm*width,
							   b+dy_norm*sl+dx_norm*width);
				path.addPolygon(pts);
				painter_.fillPath(path, col);
				break;
			case 2:
				pts << QPointF(p2x, p2y);
				pts << QPointF(c, d);
				pts << QPointF(p1x, p1y);
				sl = norm-arrowWidth;
				pts << QPointF(a+dx_norm*sl+dy_norm*width, 
							   b+dy_norm*sl-dx_norm*width);
				pts << QPointF(a+dy_norm*width, b-dx_norm*width);
				pts << QPointF(a-dy_norm*width, b+dx_norm*width);
				pts << QPointF(a+dx_norm*sl-dy_norm*width,
							   b+dy_norm*sl+dx_norm*width);
				path.addPolygon(pts);
				painter_.fillPath(path, col);
				break;
			default:
				break;
		}
		
		update();
	}

	void PaintingWindow::display(const QImage& image, int xoff, int yoff,
								               double fact)
	{
		painter_.translate(xoff, yoff);
		painter_.scale(qreal(fact), qreal(fact));
		painter_.drawImage(0, 0, image);
		painter_.scale(qreal(1./fact), qreal(1./fact));
		painter_.translate(-xoff, -yoff);
		update();
	}

	void PaintingWindow::fillCircle(int x, int y, int r, const QColor& c)
	{
		QPainterPath path;
		path.addEllipse(qreal(x)-r/2., qreal(y)-r/2., qreal(r), qreal(r));
		painter_.fillPath(path, c);
		update();
	}

  void PaintingWindow::fillCircle(const QPointF& p, qreal r, const QColor& c)
  {
    QPainterPath path;
    path.addEllipse(p, r, r);
    painter_.fillPath(path, c);
    update();
  }

	void PaintingWindow::fillEllipse(int x, int y, int w, int h, 
								                   const QColor& c)
	{
		QPainterPath path;
		path.addEllipse(qreal(x), qreal(y), qreal(w), qreal(h));
		painter_.fillPath(path, c);
		update();
	}

  void PaintingWindow::fillEllipse(const QPointF& p, qreal rx, qreal ry,
                                   qreal degree, const QColor& c)
  {
    painter_.save();
    painter_.translate(p);
    painter_.rotate(degree);
    painter_.translate(-rx, -ry);
    QPainterPath path;
    path.addEllipse(0., 0., 2*rx, 2*ry);
    painter_.fillPath(path, c);
    painter_.restore();
    update();
  }

	void PaintingWindow::fillPoly(const QPolygonF& polygon, const QColor& c)
	{
		QPainterPath path;
		path.addPolygon(polygon);
		painter_.fillPath(path, c);
		update();
	}

	void PaintingWindow::fillRect(int x, int y, int w, int h, 
								                const QColor& c)
	{
		painter_.setPen(c);
		painter_.fillRect(x, y, w, h, c);
		update();
	}

	void PaintingWindow::clear()
	{
		pixmap_.fill();
		update();
	}

  void PaintingWindow::setAntialiasing(bool on)
  { painter_.setRenderHints(QPainter::Antialiasing, on); }

  void PaintingWindow::setTransparency(bool on)
  {
    if (on)
      painter_.setCompositionMode(QPainter::CompositionMode_Multiply);
    else
      painter_.setCompositionMode(QPainter::CompositionMode_Source);
  }

  void PaintingWindow::saveScreen(const QString& filename)
  {
    pixmap_.save(filename);
  }

	void PaintingWindow::waitForEvent(int ms)
	{
		event_listening_timer_.setInterval(ms);
		event_listening_timer_.start();
	}

	void PaintingWindow::eventListeningTimerStopped()
	{
		emit sendEvent(noEvent());
	}

	void PaintingWindow::mouseMoveEvent(QMouseEvent *event)
	{
		emit(event->x(), event->y(), event->buttons());

		if (event_listening_timer_.isActive())
		{
			event_listening_timer_.stop();
			sendEvent(mouseMoved(event->x(), event->y(), event->buttons(),
				        event->modifiers()));
		}
	}

	void PaintingWindow::mousePressEvent(QMouseEvent *event)
	{
#ifdef Q_OS_MAC
		Qt::MouseButtons buttons = (event->modifiers() == Qt::ControlModifier &&
									event->buttons() == Qt::LeftButton) ? 
		Qt::MiddleButton : event->buttons();
		emit pressedMouseButtons(event->x(), event->y(), buttons);
#else
		emit pressedMouseButtons(event->x(), event->y(), event->buttons());
#endif
		if (event_listening_timer_.isActive())
		{
			event_listening_timer_.stop();
			sendEvent(mousePressed(event->x(), event->y(), event->buttons(), 
				        event->modifiers()));
		}
	}

	void PaintingWindow::mouseReleaseEvent(QMouseEvent *event)
	{
		//qDebug() << "Released " << event->pos().x() << " " << event->pos().y();
#ifdef Q_OS_MAC
		Qt::MouseButtons buttons = (event->modifiers() == Qt::ControlModifier &&
			event->button() == Qt::LeftButton) ? 
			Qt::MiddleButton : event->button();
		emit releasedMouseButtons(event->x(), event->y(), buttons);
#else
    //qDebug() << int(event->button());
		emit releasedMouseButtons(event->x(), event->y(), event->button());
#endif
		if (event_listening_timer_.isActive())
		{
			event_listening_timer_.stop();
			sendEvent(mouseReleased(event->x(), event->y(), 
					                    event->button(), event->modifiers()));
		}
	}

  void PaintingWindow::keyPressEvent(QKeyEvent *event)
  {
    emit pressedKey(event->key());
    if (event_listening_timer_.isActive())
    {
      event_listening_timer_.stop();
      sendEvent(keyPressed(event->key(), event->modifiers()));
    }
  }

  void PaintingWindow::keyReleaseEvent(QKeyEvent *event)
  {
    emit releasedKey(event->key());
    if (event_listening_timer_.isActive())
    {
      event_listening_timer_.stop();
      sendEvent(keyReleased(event->key(), event->modifiers()));
    }
  }

	void PaintingWindow::paintEvent(QPaintEvent *event)
	{		
		QPainter p(this);
		p.drawPixmap(0, 0, pixmap_);
	}

} /* namespace DO */