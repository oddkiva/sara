#pragma once

#include <QQuaternion>
#include <QTime>
#include <QTimer>
#include <QVector>
#include <QVector3D>


namespace DO { namespace Kalpana {

  class TrackBall
  {
  public:
    TrackBall();

    // Coordinates in [-1,1]x[-1,1]
    void push(const QPointF& p, const QQuaternion& transformation);

    void move(const QPointF& p);

    void release(const QPointF& p);

    QQuaternion rotation() const;

  private:
    QQuaternion m_rotation;
    QVector3D m_axis;
    QPointF m_lastPos;
    bool m_pressed;
  };

} /* namespace Kalpana */
} /* namespace DO */
