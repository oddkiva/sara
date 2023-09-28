// ========================================================================== //
// This file is part of Kalpana.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <QOpenGLWidget>
#include <QtDebug>
#include <QtOpenGL>

#include <DO/Kalpana/Qt/2D.hpp>


namespace DO { namespace Kalpana {

  Canvas::Canvas(QWidget* parent)
    : QGraphicsView{parent}
  {
    setViewport(new QOpenGLWidget);
    setTransformationAnchor(AnchorUnderMouse);
    setRenderHints(QPainter::Antialiasing);
    setDragMode(RubberBandDrag);
    setMouseTracking(true);

    setScene(new QGraphicsScene);
  }

  void Canvas::plot(const VectorXd& x, const VectorXd& y, const QPen& pen)
  {
    scene()->addItem(new Graph{x, y, pen});
    fitInView(sceneRect());
  }

  void Canvas::drawForeground(QPainter* painter, const QRectF& rect)
  {
    Q_UNUSED(rect);

    painter->resetTransform();
    const auto w = viewport()->width();
    const auto h = viewport()->height();

    const auto padding = QPoint{20, 20};
    painter->drawRect(QRectF{padding, QPoint{w, h} - padding});
  }

  void Canvas::keyPressEvent(QKeyEvent* event)
  {
    // Adjust view.
    if (event->key() == Qt::Key_F)
      fitInView(sceneRect());

    QGraphicsView::keyPressEvent(event);
  }

  void Canvas::mousePressEvent(QMouseEvent* event)
  {
    QGraphicsView::mousePressEvent(event);
  }

  void Canvas::mouseReleaseEvent(QMouseEvent* event)
  {
    QGraphicsView::mouseReleaseEvent(event);
  }

  void Canvas::mouseMoveEvent(QMouseEvent* event)
  {
    auto point = mapToScene(event->pos());
    qDebug() << "Canvas size" << size();
    qDebug() << "view coordinates" << event->pos();
    qDebug() << "scene coordinates" << point;

    QGraphicsView::mouseMoveEvent(event);
  }

  void Canvas::wheelEvent(QWheelEvent* event)
  {
    if (event->modifiers() == Qt::ControlModifier)
      scaleView(pow(double(2), event->angleDelta().y() / 240.0));
    QGraphicsView::wheelEvent(event);
  }

  void Canvas::resizeEvent(QResizeEvent* event)
  {
    QGraphicsView::resizeEvent(event);
  }

  void Canvas::scaleView(qreal scaleFactor)
  {
    scale(scaleFactor, scaleFactor);
  }

}}  // namespace DO::Kalpana
