#pragma once

#include <DO/Sara/Defines.hpp>

#include <Eigen/Geometry>

namespace DO::Kalpana::GL {

  class DO_SARA_EXPORT TrackBall
  {
  public:
    inline TrackBall() = default;

    // Coordinates in [-1,1]x[-1,1]
    auto push(const Eigen::Vector2f& p) -> void;

    auto move(const Eigen::Vector2f& p) -> void;

    auto release(const Eigen::Vector2f& p) -> void;

    auto rotation() const -> Eigen::Quaternionf;

  private:
    Eigen::Quaternionf _rotation = Eigen::Quaternionf::Identity();
    Eigen::Vector3f _axis = Eigen::Vector3f::UnitY();
    Eigen::Vector2f _last_pos = Eigen::Vector2f::Zero();
    bool _pressed = false;
  };

}  // namespace DO::Kalpana::GL
