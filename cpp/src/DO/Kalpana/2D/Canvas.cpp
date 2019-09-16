#include <QtDebug>
#include <QtOpenGL>

#include <DO/Kalpana/2D.hpp>


namespace DO { namespace Kalpana {

  Canvas::Canvas(QWidget *parent)
    : QGraphicsView{ parent }
  {
    setViewport(new QGLWidget(QGLFormat(QGL::SampleBuffers)));
    setTransformationAnchor(AnchorUnderMouse);
    setRenderHints(QPainter::Antialiasing);
    setDragMode(RubberBandDrag);
    setMouseTracking(true);

    setScene(new QGraphicsScene);
  }

  void Canvas::plot(const VectorXd& x, const VectorXd& y, const QPen& pen)
  {
    scene()->addItem(new Graph{ x, y, pen });
    fitInView(sceneRect());
  }

  void Canvas::drawForeground(QPainter *painter, const QRectF& rect)
  {
    Q_UNUSED(rect);

    painter->resetTransform();
    const auto w = viewport()->width();
    const auto h = viewport()->height();

    const auto padding = QPoint{ 20, 20 };
    painter->drawRect(QRectF{ padding, QPoint{ w, h } - padding });
  }

  void Canvas::keyPressEvent(QKeyEvent *event)
  {
    // Adjust view.
    if (event->key() == Qt::Key_F)
      fitInView(sceneRect());

    QGraphicsView::keyPressEvent(event);
  }

  void Canvas::mousePressEvent(QMouseEvent *event)
  {
    QGraphicsView::mousePressEvent(event);
  }

  void Canvas::mouseReleaseEvent(QMouseEvent *event)
  {
    QGraphicsView::mouseReleaseEvent(event);
  }

  void Canvas::mouseMoveEvent(QMouseEvent *event)
  {
    auto point = mapToScene(event->pos());
    qDebug() << "Canvas size" << size();
    qDebug() << "view coordinates" << event->pos();
    qDebug() << "scene coordinates" << point;

    QGraphicsView::mouseMoveEvent(event);
  }

  void Canvas::wheelEvent(QWheelEvent *event)
  {
    if (event->modifiers() == Qt::ControlModifier)
      scaleView(pow(double(2), event->delta() / 240.0));
    QGraphicsView::wheelEvent(event);
  }

  void Canvas::resizeEvent(QResizeEvent *event)
  {
    QGraphicsView::resizeEvent(event);
  }

  void Canvas::scaleView(qreal scaleFactor)
  {
    scale(scaleFactor, scaleFactor);
  }

} /* namespace Kalpana */
} /* namespace DO */
