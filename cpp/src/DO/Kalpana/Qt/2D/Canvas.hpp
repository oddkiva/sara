// ========================================================================== //
// This file is part of Kalpana.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_KALPANA_2D_CANVAS_HPP
#define DO_KALPANA_2D_CANVAS_HPP

#include <DO/Sara/Defines.hpp>

#include <QGraphicsView>

#include <Eigen/Core>


namespace DO { namespace Kalpana {

  using namespace Eigen;


  class DO_SARA_EXPORT Canvas : public QGraphicsView
  {
  public:
    Canvas(QWidget* parent = 0);

  public slots:
    void plot(const VectorXd& x, const VectorXd& y, const QPen& pen = QPen{});

  protected:
    void drawForeground(QPainter* event, const QRectF& rect);

    void keyPressEvent(QKeyEvent* event);

    void mousePressEvent(QMouseEvent* event);

    void mouseReleaseEvent(QMouseEvent* event);

    void mouseMoveEvent(QMouseEvent* event);

    void wheelEvent(QWheelEvent* event);

    void resizeEvent(QResizeEvent* event);

  private:
    void scaleView(qreal scaleFactor);
  };

} /* namespace Kalpana */
} /* namespace DO */

#endif /* DO_KALPANA_2D_CANVAS_HPP */
