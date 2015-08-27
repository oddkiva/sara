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

#include <QtGui>
#include <QGraphicsView>
#include <QStyleOptionGraphicsItem>

#include <DO/Sara/Graphics/DerivedQObjects/PixmapItem.hpp>


namespace DO { namespace Sara {

  void GraphicsPixmapItem::keyPressEvent(QKeyEvent *event)
  {
    if (!isSelected())
      return;

    if (event->key() == Qt::Key_Plus)
    {
      m_scaleFactor *= 1.1;
      setScale(m_scaleFactor);
    }
    else if (event->key() == Qt::Key_Minus)
    {
      m_scaleFactor /= 1.1;
      setScale(m_scaleFactor);
    }
  }

} /* namespace Sara */
} /* namespace DO */
