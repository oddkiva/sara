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

#ifndef DO_SARA_GRAPHICS_PIXMAPITEM_HPP
#define DO_SARA_GRAPHICS_PIXMAPITEM_HPP

#include <QGraphicsPixmapItem>


namespace DO { namespace Sara {

  class ImageItem : public QGraphicsPixmapItem
  {
  public:
    ImageItem(QGraphicsItem* parent = 0) : QGraphicsPixmapItem(parent) {}
    ImageItem(const QPixmap& pixmap, QGraphicsItem* parent = 0)
      : QGraphicsPixmapItem(pixmap, parent)
      , scaleFactor(1.0)
    {
      setFlags(ItemIsMovable | ItemIsSelectable | ItemIsFocusable);
    }

  protected:
    void keyPressEvent(QKeyEvent *event);

  private:
    qreal scaleFactor;
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GRAPHICS2_PIXMAPITEM_HPP */
