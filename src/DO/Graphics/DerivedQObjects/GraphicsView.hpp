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

//! @file

#ifndef DO_GRAPHICS_GRAPHICSVIEW_HPP
#define DO_GRAPHICS_GRAPHICSVIEW_HPP

#include "../Events.hpp"
#include <QGraphicsView>
#include <QTimer>

namespace DO {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  //! \brief QGraphicsView-derived class used to view interactively images.
  class GraphicsView : public QGraphicsView
  {
    Q_OBJECT
  public:
    GraphicsView(int width, int height, 
                 const QString& windowTitle = "DO++",
                 int x = -1, int y = -1,
                 QWidget* parent = 0);
    void activateOpenGL();
    
    QGraphicsItem *lastAddedItem() { return last_inserted_item_; }

  public slots:
    void addItem(QGraphicsItem *item, QGraphicsItem *parent = 0);
    void addImageItem(const QImage& image, bool randomPos = false);
    void drawPoint(int x, int y, const QColor& c, QGraphicsPixmapItem *item);

  protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *);
    void keyPressEvent(QKeyEvent *event);
    void closeEvent(QCloseEvent *event);

  signals:
    void pressedMouseButtons(int x, int y, Qt::MouseButtons buttons);
    void releasedMouseButtons(int x, int y, Qt::MouseButtons buttons);
    void pressedKey(int key);
    void releasedKey(int key);
    void sendEvent(Event e);

  private:
    void scaleView(qreal scaleFactor);

  private:
    QTimer event_listening_timer_;
    QGraphicsItem *last_inserted_item_;
  };

  //! @}
} /* namespace DO */

#endif /* DO_GRAPHICS_GRAPHICSVIEW_HPP */