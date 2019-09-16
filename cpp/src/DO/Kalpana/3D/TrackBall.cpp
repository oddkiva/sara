#include <DO/Kalpana/3D.hpp>


namespace DO { namespace Kalpana {

  TrackBall::TrackBall()
  {
    m_pressed = false;
    m_axis = QVector3D(0, 1, 0);
    m_rotation = QQuaternion();
  }

  void TrackBall::push(const QPointF& p, const QQuaternion &)
  {
    m_rotation = rotation();
    m_pressed = true;
    m_lastPos = p;
  }

  static void projectToSphere(QVector3D& x)
  {
    const auto sqrZ = 1 - QVector3D::dotProduct(x, x);
    if (sqrZ > 0)
      x.setZ(std::sqrt(sqrZ));
    else
      x.normalize();
  }

  void TrackBall::move(const QPointF& p)
  {
    if (!m_pressed)
      return;

    // Get the last position and project it on the sphere
    auto lastPos3D = QVector3D(m_lastPos.x(), m_lastPos.y(), 0.0f);
    projectToSphere(lastPos3D);

    // Get the current position and project it on the sphere
    QVector3D currentPos3D = QVector3D(p.x(), p.y(), 0.0f);
    projectToSphere(currentPos3D);

    // Compute the new axis by cross product
    m_axis = QVector3D::crossProduct(lastPos3D, currentPos3D);
    m_axis.normalize();

    // Compose the old rotation with the new rotation.
    // Remember that quaternions do not commute.
#ifdef __linux__
    m_rotation = QQuaternion::fromAxisAndAngle(m_axis, 1.0) * m_rotation;
#else
    m_rotation = QQuaternion::fromAxisAndAngle(m_axis, 2.0) * m_rotation;
#endif

    // Remember the current position as the last position when move is called again.
    m_lastPos = p;
  }

  void TrackBall::release(const QPointF& p)
  {
    move(p);
    m_pressed = false;
  }

  QQuaternion TrackBall::rotation() const
  {
    if (m_pressed)
      return m_rotation;
    return  QQuaternion::fromAxisAndAngle(m_axis, 2.0) * m_rotation;
  }

} /* namespace Kalpana */
} /* namespace DO */
