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

#ifndef DO_GRAPHICS_PIXMAPITEM_HPP
#define DO_GRAPHICS_PIXMAPITEM_HPP

#include <QGraphicsPixmapItem>

namespace DO {

  class PixmapItem : public QGraphicsPixmapItem
  {
  public:
    PixmapItem(QGraphicsItem* parent = 0) : QGraphicsPixmapItem(parent) {}
    PixmapItem(const QPixmap& pixmap, QGraphicsItem* parent = 0)
      : QGraphicsPixmapItem(pixmap, parent)
      , scaleFactor(1.0)
      , childrenHidden(true)
    {
      setFlags(ItemIsMovable | ItemIsSelectable | ItemIsFocusable);
    }

  protected:
    void keyPressEvent(QKeyEvent *event);

  private:
    qreal scaleFactor;
    bool childrenHidden;
  };

} /* namespace DO */

#endif /* DO_GRAPHICS2_PIXMAPITEM_HPP */