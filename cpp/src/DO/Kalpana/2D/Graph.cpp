#include <QGraphicsItem>

#include <DO/Kalpana/2D.hpp>


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
