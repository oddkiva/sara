#pragma once

#include <QPen>


namespace plt {

  inline
  QPen style(const QColor& color, double width, const QString& style)
  {
    auto pen = QPen{ color, width };
    if (style == "-")
      pen.setStyle(Qt::SolidLine);
    if (style == "--")
      pen.setStyle(Qt::DashLine);
    if (style == "-.")
      pen.setStyle(Qt::DashDotLine);
    return pen;
  }

}
