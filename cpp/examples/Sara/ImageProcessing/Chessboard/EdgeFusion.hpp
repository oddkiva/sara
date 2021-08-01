#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Sara/DisjointSets.hpp>
#include <DO/Sara/ImageProcessing/EdgeGrouping.hpp>


namespace DO::Sara {

  struct EdgeGraph
  {
    const EdgeAttributes& attributes;
    const std::vector<std::vector<Eigen::Vector2i>>& edge_chains;
    const std::vector<Eigen::Vector2f>& mean_gradients;
    std::vector<std::size_t> edge_ids;

    Tensor_<std::uint8_t, 2> A;

    EdgeGraph(const EdgeAttributes& attrs,
              const std::vector<std::vector<Eigen::Vector2i>>& edges,
              const std::vector<Eigen::Vector2f>& mean_grads)
      : attributes{attrs}
      , edge_chains{edges}
      , mean_gradients{mean_grads}
    {
      initialize();
    }

    auto initialize() -> void
    {
      const auto n = static_cast<std::int32_t>(attributes.edges.size());
      std::iota(edge_ids.begin(), edge_ids.end(), 0u);

      A.resize(n, n);
      A.flat_array().fill(0);
      for (auto i = 0; i < n; ++i)
      {
        for (auto j = i; j < n; ++j)
        {
          if (i == j)
            A(i, j) = 1;
          else if (edge(i).size() >= 2 && edge(j).size() >= 2)
            A(i, j) = is_aligned(i, j);
          A(j, i) = A(i, j);
        }
      }
    }

    auto edge(std::size_t i) const -> const Edge<double>&
    {
      return attributes.edges[i];
    }

    auto center(std::size_t i) const -> const Eigen::Vector2d&
    {
      return attributes.centers[i];
    }

    auto axes(std::size_t i) const -> const Eigen::Matrix2d&
    {
      return attributes.axes[i];
    }

    auto lengths(std::size_t i) const -> const Eigen::Vector2d&
    {
      return attributes.lengths[i];
    }

    auto rect(std::size_t i) const -> OrientedBox
    {
      return {center(i), axes(i), lengths(i)};
    }

    auto is_aligned(std::size_t i, std::size_t j) const -> bool
    {
      // const auto& ri = rect(i);
      // const auto& rj = rect(j);
      // const auto& di = ri.axes.col(0);
      // const auto& dj = rj.axes.col(0);

      const auto& di = mean_gradients[i];
      const auto& dj = mean_gradients[j];

      // const auto& ai = edge_chains[i].front();
      // const auto& bi = edge_chains[i].back();

      // const auto& aj = edge_chains[j].front();
      // const auto& bj = edge_chains[j].back();

      const auto& ai = attributes.edges[i].front();
      const auto& bi = attributes.edges[i].back();

      const auto& aj = attributes.edges[j].front();
      const auto& bj = attributes.edges[j].back();

      constexpr auto dist_thres = 10.f;
      constexpr auto dist_thres_2 = dist_thres * dist_thres;
#ifdef ANTISYMMETRIC_ADJACENCE
      const auto is_adjacent = //
        // [bi, ai] ~ [aj, bj]
        ((ai - aj).squaredNorm() < dist_thres_2 && (bi - bj).squaredNorm() >= dist_thres_2) ||
        // [bi, ai] ~ [bj, aj]
        ((ai - bj).squaredNorm() < dist_thres_2 && (bi - aj).squaredNorm() >= dist_thres_2) ||
        // [ai, bi] ~ [aj, bj]
        ((bi - aj).squaredNorm() < dist_thres_2 && (ai - bj).squaredNorm() >= dist_thres_2) ||
        // [ai, bi] ~ [bj, aj]
        ((bi - bj).squaredNorm() < dist_thres_2 && (ai - aj).squaredNorm() >= dist_thres_2);
#else
      const auto is_adjacent = (ai - aj).squaredNorm() < dist_thres_2 ||
                               (ai - bj).squaredNorm() < dist_thres_2 ||
                               (bi - aj).squaredNorm() < dist_thres_2 ||
                               (bi - bj).squaredNorm() < dist_thres_2;
#endif

      // return is_adjacent && std::abs(di.dot(dj)) > std::cos(30._deg);
      return is_adjacent && di.dot(dj) > std::cos(20._deg);
    }

    auto group_by_alignment() const
    {
      const auto n = A.rows();

      auto ds = DisjointSets(n);

      for (auto i = 0; i < n; ++i)
        ds.make_set(i);

      for (auto i = 0; i < n; ++i)
        for (auto j = i; j < n; ++j)
          if (A(i, j) == 1)
            ds.join(ds.node(i), ds.node(j));

      auto groups = std::map<std::size_t, std::vector<std::size_t>>{};
      for (auto i = 0; i < n; ++i)
      {
        const auto c = ds.component(i);
        groups[c].push_back(i);
      }

      return groups;
    }
  };

  auto check_edge_grouping(                                          //
      const ImageView<Rgb8>& frame,                                  //
      const std::vector<Edge<double>>& edges_refined,                //
      const std::vector<Edge<int>>& edge_chains,                     //
      const std::vector<Eigen::Vector2f>& mean_gradients,            //
      const std::vector<Eigen::Vector2d>& centers,                   //
      const std::vector<Eigen::Matrix2d>& axes,                      //
      const std::vector<Eigen::Vector2d>& lengths)                   //
      -> void;


}  // namespace DO::Sara
