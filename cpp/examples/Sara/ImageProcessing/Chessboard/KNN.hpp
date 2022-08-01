#pragma once

#include "Corner.hpp"


template <typename T>
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
      const auto d_uv = (pu - pv).squaredNorm();

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

template <typename V>
struct KnnGraph
{
  int _k;
  std::vector<V> _vertices;
  Eigen::MatrixXi _neighbors;
  Eigen::MatrixXf _distances;
  Eigen::MatrixXf _affinities;
  Eigen::MatrixXf _circular_profiles;
  Eigen::MatrixXf _affinity_scores;
  Eigen::VectorXf _unary_scores;

  struct VertexScore
  {
    std::size_t vertex;
    float score;
    inline auto operator<(const VertexScore& other) const
    {
      return score < other.score;
    }
  };


  inline auto vertex(int v) const -> const V&
  {
    return _vertices[v];
  }

  inline auto nearest_neighbor(const int v, const int k) const -> const V&
  {
    return _vertices[_neighbors(k, v)];
  };

  inline auto compute_affinity_scores() -> void
  {
    const auto n = _vertices.size();
    const auto k = _neighbors.rows();

    _affinity_scores.resize(k, n);
    _unary_scores.resize(n);

    for (auto u = 0u; u < n; ++u)
    {
      const auto fu = _circular_profiles.col(u);

      for (auto nn = 0; nn < k; ++nn)
      {
        const auto v = _neighbors(nn, u);
        static constexpr auto undefined_neighbor = -1;
        if (v == undefined_neighbor)
        {
          _affinity_scores(nn, u) = -std::numeric_limits<float>::max();
          continue;
        }
        const auto fv = _circular_profiles.col(v);

        auto affinities = Eigen::Matrix4f{};
        for (auto i = 0; i < fu.size(); ++i)
          for (auto j = 0; j < fv.size(); ++j)
            affinities(i, j) = std::abs(dir(fu(i)).dot(dir(fv(j))));

        const Eigen::RowVector4f best_affinities =
            affinities.colwise().maxCoeff();
        _affinity_scores(nn, u) = best_affinities.sum();
      }

      _unary_scores(u) = _affinity_scores.col(u).sum();
    }
  }
};
