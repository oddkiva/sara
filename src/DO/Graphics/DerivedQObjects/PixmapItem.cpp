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

#include "PixmapItem.hpp"
#include <QtGui>
#include <QGraphicsView>
#include <QStyleOptionGraphicsItem>

namespace DO {
    
  void PixmapItem::keyPressEvent(QKeyEvent *event)
  {
    if (!isSelected())
      return;

    if (event->key() == Qt::Key_Plus) {
      scaleFactor *= 1.1;
      setScale(scaleFactor);
    } else if (event->key() == Qt::Key_Minus) {
      scaleFactor /= 1.1;
      setScale(scaleFactor);
    }/* else if (event->key() == Qt::Key_H)
     {
     childrenHidden = !childrenHidden;
     for (QList<QGraphicsItem *>::iterator item = childItems().begin(); 
     item != childItems().end(); ++item)
     (*item)->setVisible(childrenHidden);
     }*/

    //QGraphicsPixmapItem::keyPressEvent(event);
  }

} /* namespace DO */
