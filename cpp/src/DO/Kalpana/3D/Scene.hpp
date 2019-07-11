#pragma once

#include <memory>
#include <vector>

#include <DO/Kalpana/3D/SceneItem.hpp>


namespace DO { namespace Kalpana {

  class Scene
  {
  public:
    Scene() = default;

    PointCloud * scatter(const std::vector<Vector3f>& points);

    std::vector<std::unique_ptr<SceneItem>> _objects{};
  };

} /* namespace Kalpana */
} /* namespace DO */
