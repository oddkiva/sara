#pragma once

#include <DO/Sara/Core/Image/Image.hpp>


namespace DO::Sara {

  inline auto extract_zero_level_set(const Image<float>& phi)
  {
    auto zeros = std::vector<Eigen::Vector2i>{};
    zeros.reserve(phi.sizes().maxCoeff());

    for (auto y = 1; y < phi.height() - 1; ++y)
    {
      for (auto x = 1; x < phi.width() - 1; ++x)
      {
        const auto sx = phi(x - 1, y) * phi(x + 1, y);
        const auto sy = phi(x, y - 1) * phi(x, y + 1);
        if (sx < 0 || sy < 0)
          zeros.emplace_back(x, y);
      }
    }

    return zeros;
  }

}  // namespace DO::Sara
