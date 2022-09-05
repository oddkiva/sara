// ========================================================================== //
// This file is part of Kalpana.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <QGraphicsItem>

#include <DO/Kalpana/Qt/2D.hpp>


namespace DO { namespace Kalpana {

  void Graph::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                   QWidget *widget)
  {
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->setPen(m_pen);
    painter->setOpacity(0.5);

    auto path = QPainterPath{};
    path.moveTo(polygon().front());
    for (int i = 1; i < polygon().size(); ++i)
      path.lineTo(polygon()[i]);

    painter->drawPath(path);
  }

} /* namespace Kalpana */
} /* namespace DO */
