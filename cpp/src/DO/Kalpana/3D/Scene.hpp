#pragma once

#include <DO/Kalpana/Defines.hpp>

#include <DO/Kalpana/3D/SceneItem.hpp>

#include <memory>
#include <vector>


namespace DO { namespace Kalpana {

  class DO_KALPANA_EXPORT Scene
  {
  public:
    Scene() = default;

    Scene(const Scene&) = delete;

    Scene(Scene&& other)
      : _objects{std::move(other._objects)}
    {
    }

    PointCloud* scatter(const std::vector<Vector3f>& points);

    std::vector<std::unique_ptr<SceneItem>> _objects{};
  };

}}  // namespace DO::Kalpana
