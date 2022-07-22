#include "Corner.hpp"

template <typename T, Eigen::Index N>
auto k_nearest_neighbors(const std::vector<Corner<T>>& points, const int k)
    -> std::pair<Eigen::MatrixXi, Eigen::MatrixXf>
{
  const auto n = points.size();

  auto neighbors = Eigen::MatrixXi{k, n};
  auto distances = Eigen::MatrixXf{k, n};
  neighbors.setConstant(-1);
  distances.setConstant(std::numeric_limits<float>::infinity());

  for (auto u = 0u; u < n; ++u)
  {
    const auto& pu = points[u].position();

    for (auto v = 0u; v < n; ++v)
    {
      if (v == u)
        continue;

      const auto& pv = points[v].position();
      const auto d_uv = static_cast<float>((pu - pv).squaredNorm());

      for (auto a = 0; a < k; ++a)
      {
        if (d_uv < distances(a, u))
        {
          // Shift the neighbors and distances.
          auto neighbor = static_cast<int>(v);
          auto dist = d_uv;
          for (auto b = a; b < k; ++b)
          {
            std::swap(neighbor, neighbors(b, u));
            std::swap(dist, distances(b, u));
          }

          break;
        }
      }
    }
  }

  return std::make_pair(neighbors, distances);
}
