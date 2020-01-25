#ifndef DO_KALPANA_2D_GRAPH_HPP
#define DO_KALPANA_2D_GRAPH_HPP

#include <QGraphicsItem>

#include <Eigen/Core>


namespace DO { namespace Kalpana {

  using namespace Eigen;

  class Graph : public QGraphicsPolygonItem
  {
  public:
    Graph(const QPolygonF& points)
      : QGraphicsPolygonItem{ points }
    {
    }

    Graph(const VectorXd& X, const VectorXd& Y, const QPen& pen = QPen{})
      : QGraphicsPolygonItem{}
      , m_pen{ pen }
    {
      m_pen.setCosmetic(true);

      auto poly = QPolygonF(X.size());
      for (int i = 0; i < X.size(); ++i)
        poly[i] = QPointF{ X[i], -Y[i] };

      setPolygon(std::move(poly));
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget);

  private:
    QPen m_pen;
  };

} /* namespace Kalpana */
} /* namespace DO */


#endif /* DO_KALPANA_2D_GRAPH_HPP */
