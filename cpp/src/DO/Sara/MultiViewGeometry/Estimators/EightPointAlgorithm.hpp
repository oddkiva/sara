#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  struct EightPointAlgorithm
  {
    template <typename Sample>
    Matrix3d operator()(const Sample&)
    {
      return Matrix3d{};
    }
  };


} /* namespace Sara */
} /* namespace DO */
