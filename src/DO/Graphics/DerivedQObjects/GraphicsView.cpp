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

#include "GraphicsView.hpp"
#include "PixmapItem.hpp"
#include <QtGui>
#include <QtOpenGL>

namespace DO {
    
  GraphicsView::GraphicsView(int w, int h, const QString& windowTitle,
                             int x, int y, QWidget* parent)
    : QGraphicsView(new QGraphicsScene, parent)
  {
    //activateOpenGL();
    setAttribute(Qt::WA_DeleteOnClose);
    setTransformationAnchor(AnchorUnderMouse);
    setRenderHints(QPainter::Antialiasing);
    setDragMode(RubberBandDrag);

    setWindowTitle(windowTitle);    
    resize(w, h);
    move(x, y);
    show();
  }

  void GraphicsView::activateOpenGL()
  {
    setViewport( new QGLWidget(QGLFormat(QGL::SampleBuffers)) );
  }

  void GraphicsView::addItem(QGraphicsItem *item, QGraphicsItem *parent)
  {
    scene()->addItem(item);
    last_inserted_item_ = item;
    if (parent)
      last_inserted_item_->setParentItem(parent);
  }

  void GraphicsView::addImageItem(const QImage& image, bool randomPos)
  {
    last_inserted_item_ = new PixmapItem(QPixmap::fromImage(image));
    addItem(last_inserted_item_);
    if (randomPos)
      last_inserted_item_->setPos(QPointF(qrand()%10240, qrand()%7680));
  }

  void GraphicsView::drawPointOnPixmapItem(int x, int y, const QColor& c,
                                           QGraphicsPixmapItem *pixItem)
  {
    QPixmap pixmap(pixItem->pixmap());
    QPainter p(&pixmap);
    p.setPen(c);
    p.drawPoint(x, y);
    pixItem->setPixmap(pixmap);
  }

  void GraphicsView::mousePressEvent(QMouseEvent *event)
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
    // Handle the mouse press event as usual.
    QGraphicsView::mousePressEvent(event);
  }

  void GraphicsView::mouseReleaseEvent(QMouseEvent *event)
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
    // Handle the mouse release event as usual.
    QGraphicsView::mouseReleaseEvent(event);
  }

  void GraphicsView::wheelEvent(QWheelEvent *event)
  {
    if (event->modifiers() == Qt::ControlModifier)
    {
      scaleView(pow(double(2), event->delta() / 240.0));
      return;
    }
    QGraphicsView::wheelEvent(event);
  }

  void GraphicsView::keyPressEvent(QKeyEvent *event)
  {
    emit pressedKey(event->key());
    if (event_listening_timer_.isActive())
    {
      event_listening_timer_.stop();
      sendEvent(keyPressed(event->key(), event->modifiers()));
    }
    QGraphicsView::keyPressEvent(event);
  }

  void GraphicsView::closeEvent(QCloseEvent *event)
  {
    QGraphicsView::closeEvent(event);
    if(event->spontaneous())
    {
      qWarning() << "\n\nWarning: you closed a window unexpectedly!\n\n";
      qWarning() << "Graphical application is terminating...";
      qApp->exit(0);
    }
  }

  void GraphicsView::scaleView(qreal scaleFactor)
  {
    scale(scaleFactor, scaleFactor);
  }

} /* namespace DO */