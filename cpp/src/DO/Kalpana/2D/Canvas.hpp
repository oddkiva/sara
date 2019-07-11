#ifndef DO_KALPANA_2D_CANVAS_HPP
#define DO_KALPANA_2D_CANVAS_HPP

#include <QGraphicsView>

#include <Eigen/Core>


namespace DO { namespace Kalpana {

  using namespace Eigen;


  class Canvas : public QGraphicsView
  {
  public:
    Canvas(QWidget *parent = 0);

  public slots:
    void plot(const VectorXd& x, const VectorXd& y, const QPen& pen = QPen{});

  protected:
    void drawForeground(QPainter *event, const QRectF& rect);

    void keyPressEvent(QKeyEvent *event);

    void mousePressEvent(QMouseEvent *event);

    void mouseReleaseEvent(QMouseEvent *event);

    void mouseMoveEvent(QMouseEvent *event);

    void wheelEvent(QWheelEvent *event);

    void resizeEvent(QResizeEvent *event);

  private:
    void scaleView(qreal scaleFactor);
  };

} /* namespace Kalpana */
} /* namespace DO */

#endif /* DO_KALPANA_2D_CANVAS_HPP */
