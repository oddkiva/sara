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

#include <QRandomGenerator>
#include <QtGui>
#include <QtOpenGL>

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsView.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/PixmapItem.hpp>


namespace DO { namespace Sara {

  GraphicsView::GraphicsView(int w, int h, const QString& windowTitle,
                             int x, int y, QWidget* parent)
    : QGraphicsView(parent)
  {
    setScene(new QGraphicsScene(this));

    // Set event listener.
    m_eventListeningTimer.setSingleShot(true);
    connect(&m_eventListeningTimer, SIGNAL(timeout()),
            this, SLOT(eventListeningTimerStopped()));

    setAttribute(Qt::WA_DeleteOnClose);
    setTransformationAnchor(AnchorUnderMouse);
    setRenderHints(QPainter::Antialiasing);
    setDragMode(RubberBandDrag);

    setWindowTitle(windowTitle);
    resize(w, h);
    move(x, y);
    show();
  }

  void GraphicsView::addItem(QGraphicsItem *item, QGraphicsItem *parent)
  {
    scene()->addItem(item);
    m_lastInsertedItem = item;
    if (parent)
      m_lastInsertedItem->setParentItem(parent);
  }

  void GraphicsView::addPixmapItem(const QImage& image, bool randomPos)
  {
    m_lastInsertedItem = new GraphicsPixmapItem(QPixmap::fromImage(image));
    addItem(m_lastInsertedItem);
    if (randomPos)
      m_lastInsertedItem->setPos(QPointF(
          QRandomGenerator::global()->bounded(10240u),
          QRandomGenerator::global()->bounded(7680u)));
  }

  void GraphicsView::waitForEvent(int ms)
  {
    m_eventListeningTimer.setInterval(ms);
    m_eventListeningTimer.start();
  }

  void GraphicsView::eventListeningTimerStopped()
  {
    emit sendEvent(no_event());
  }

  void GraphicsView::wheelEvent(QWheelEvent *event)
  {
    if (event->modifiers() == Qt::ControlModifier)
    {
      scaleView(pow(double(2), event->angleDelta().y() / 240.0));
      return;
    }
    QGraphicsView::wheelEvent(event);
  }

  void GraphicsView::keyPressEvent(QKeyEvent *event)
  {
    emit pressedKey(event->key());
    if (m_eventListeningTimer.isActive())
    {
      m_eventListeningTimer.stop();
      emit sendEvent(key_pressed(event->key(), event->modifiers()));
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

} /* namespace Sara */
} /* namespace DO */
