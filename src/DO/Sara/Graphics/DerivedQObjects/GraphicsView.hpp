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

//! @file

#ifndef DO_SARA_GRAPHICS_GRAPHICSVIEW_HPP
#define DO_SARA_GRAPHICS_GRAPHICSVIEW_HPP

#include <QGraphicsView>
#include <QTimer>

#include <DO/Sara/Defines.hpp>

#include "../Events.hpp"


namespace DO { namespace Sara {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  //! @brief QGraphicsView-derived class used to view interactively images.
  class DO_SARA_EXPORT GraphicsView : public QGraphicsView
  {
    Q_OBJECT
  public:
    GraphicsView(int width, int height,
                 const QString& windowTitle = "Sara",
                 int x = -1, int y = -1,
                 QWidget* parent = 0);

    QGraphicsItem *lastAddedItem() { return m_lastInsertedItem; }

  public slots:
    void addItem(QGraphicsItem *item, QGraphicsItem *parent = 0);
    void addPixmapItem(const QImage& image, bool randomPos = false);
    void waitForEvent(int ms);
    void eventListeningTimerStopped();

  protected:
    void wheelEvent(QWheelEvent *);
    void keyPressEvent(QKeyEvent *event);
    void closeEvent(QCloseEvent *event);

  signals:
    void pressedKey(int key);
    void releasedKey(int key);
    void sendEvent(Event e);

  private:
    void scaleView(qreal scaleFactor);

  private:
    QTimer m_eventListeningTimer;
    QGraphicsItem *m_lastInsertedItem{ nullptr };
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GRAPHICS_GRAPHICSVIEW_HPP */
