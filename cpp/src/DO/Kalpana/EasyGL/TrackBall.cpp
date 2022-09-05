#include <DO/Kalpana/EasyGL/TrackBall.hpp>


namespace DO::Kalpana::GL {

  auto TrackBall::push(const Eigen::Vector2f& p) -> void
  {
    _rotation = rotation();
    _pressed = true;
    _last_pos = p;
  }

  static auto project_to_sphere(Eigen::Vector3f& x) -> void
  {
    const auto sqr_z = 1 - x.squaredNorm();
    if (sqr_z > 0)
      x.z() = std::sqrt(sqr_z);
    else
      x.normalize();
  }

  auto TrackBall::move(const Eigen::Vector2f& p) -> void
  {
    if (!_pressed)
      return;

    // Get the last position and project it on the sphere
    Eigen::Vector3f last_pos_3d;
    last_pos_3d << _last_pos, 0.f;
    project_to_sphere(last_pos_3d);

    // Get the current position and project it on the sphere
    Eigen::Vector3f current_pos_3d;
    current_pos_3d << p, 0.f;
    project_to_sphere(current_pos_3d);

    // Compute the new axis by cross product
    _axis = last_pos_3d.cross(current_pos_3d);
    _axis.normalize();

    // Compose the old rotation with the new rotation.
    // Remember that quaternions do not commute.
#ifdef __linux__
    _rotation = Eigen::AngleAxisf{1.f, _axis} * _rotation;
#else
    _rotation = Eigen::AngleAxisf{2.f, _axis} * _rotation;
#endif

    // Remember the current position as the last position when move is called
    // again.
    _last_pos = p;
  }

  auto TrackBall::release(const Eigen::Vector2f& p) -> void
  {
    move(p);
    _pressed = false;
  }

  auto TrackBall::rotation() const -> Eigen::Quaternionf
  {
    if (_pressed)
      return _rotation;
    return Eigen::AngleAxisf{2.f, _axis} * _rotation;
  }

}  // namespace DO::Kalpana::GL
