#pragma once

#include <DO/Kalpana/Math/Viewport.hpp>


namespace DO::Kalpana::GL {

  struct BasicScene
  {
    Viewport _viewport;
    Eigen::Matrix4f _projection;
    Eigen::Matrix4f _model_view;
  };

}  // namespace DO::Kalpana::GL
