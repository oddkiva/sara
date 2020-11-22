#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

auto shape_statistics(const std::vector<Eigen::Vector2i>& points)
{
  auto pmat = Eigen::MatrixXf(2, points.size());
  for (auto i = 0u; i < points.size(); ++i)
    pmat.col(i) = points[i].cast<float>();

  const auto X = pmat.row(0);
  const auto Y = pmat.row(1);


  const auto x_mean = X.array().sum() / points.size();
  const auto y_mean = Y.array().sum() / points.size();

  const auto x2_mean =
      X.array().square().sum() / points.size() - std::pow(x_mean, 2);
  const auto y2_mean =
      Y.array().square().sum() / points.size() - std::pow(y_mean, 2);
  const auto xy_mean =
      (X.array() * Y.array()).sum() / points.size() - x_mean * y_mean;

  auto statistics = Eigen::Matrix<float, 5, 1>{};
  statistics << x_mean, y_mean, x2_mean, y2_mean, xy_mean;
  return statistics;
}


// LINE SEGMENT DETECTION.
// When connecting pixel, also update the statistics of tne line.
// - orientation (cf. §2.5 in LSD)
// - rectangular approximation (§2.6)
// - density of aligned points (§2.8)

struct EdgeStatistics {
  Eigen::Vector2f orientation_sum = Eigen::Vector2f::Zero();
  float total_mass = {};
  Eigen::Vector2f unnormalized_center = Eigen::Vector2f::Zero();
  Eigen::Vector3f unnormalized_inertia = Eigen::Vector2f::Zero();

  std::vector<Eigen::Vector2i> points;

  auto angle() const -> float
  {
    return std::atan2(orientation_sum.y(), orientation_sum.x());
  }

  auto center() const -> Eigen::Vector2f
  {
    return unnormalized_center / total_mass;
  }

  auto inertia() const -> Eigen::Vector3f {
    const auto& c = center();
    const auto& m = unnormalized_inertia;
    auto m1 = Eigen::Vector3f{};
    m1(0) = m(0) * m(0) - 2 * c.x() * m(0) + c.x() * c.x();
    m1(1) = m(1) * m(1) - 2 * c.y() * m(1) + c.y() * c.y();
    m1(2) = m(0) * m(1) - c.x() * m(1) - m(0) * c.y() + c.x() * c.y();
    m1 /=  total_mass;
    return m1;
  }

  auto rectangle() const -> float {
    return {};
  }

  auto density() const -> float;
};

struct CoordsValue
{
  Eigen::Vector2i coords;
  float value;

  auto operator<(const CoordsValue& other) const
  {
    return value > other.value;
  }
};


inline auto group_line_segments(const ImageView<std::uint8_t>& edges,
                                const ImageView<float>& mag,
                                const ImageView<float>& ori,
                                float angular_threshold)
{
  const auto index = [&edges](const Eigen::Vector2i& p) {
    return p.y() * edges.width() + p.x();
  };

  const auto is_edgel = [&edges](const Eigen::Vector2i& p) {
    return edges(p) == 255;
  };

  const auto vec = [](float o) {
    return Eigen::Vector2f{cos(o), sin(o)};
  };

  const auto angular_distance = [](const auto& a, const auto& b) {
    const auto c = a.dot(b);
    const auto s = a.homogeneous().cross(b.homogeneous())(2);
    const auto dist = std::abs(std::atan2(s, c));
    return dist;
  };

  auto ds = DisjointSets(edges.size());
  auto visited = Image<std::uint8_t>{edges.sizes()};
  visited.flat_array().fill(0);

  auto statistics = std::vector<EdgeStatistics>(edges.size());

  // Collect the edgels and make as many sets as pixels.
  auto q = std::queue<Eigen::Vector2i>{};
  for (auto y = 0; y < edges.height(); ++y)
  {
    for (auto x = 0; x < edges.width(); ++x)
    {
      ds.make_set(index({x, y}));
      if (is_edgel({x, y}))
      {
        q.emplace(x, y);

        statistics[index({x, y})].total_mass = mag(x, y);

        statistics[index({x, y})].orientation_sum(0) += std::cos(ori(x, y));
        statistics[index({x, y})].orientation_sum(1) += std::sin(ori(x, y));

        statistics[index({x, y})].unnormalized_center(0) += x * mag(x, y);
        statistics[index({x, y})].unnormalized_center(1) += y * mag(x, y);

        statistics[index({x, y})].unnormalized_inertia(0) += x * x * mag(x, y);
        statistics[index({x, y})].unnormalized_inertia(1) += x * y * mag(x, y);
        statistics[index({x, y})].unnormalized_inertia(2) += y * y * mag(x, y);
      }
    }
  }

  // Neighborhood defined by 8-connectivity.
  const auto dir = std::array<Eigen::Vector2i, 8>{
      Eigen::Vector2i{1, 0},    //
      Eigen::Vector2i{1, 1},    //
      Eigen::Vector2i{0, 1},    //
      Eigen::Vector2i{-1, 1},   //
      Eigen::Vector2i{-1, 0},   //
      Eigen::Vector2i{-1, -1},  //
      Eigen::Vector2i{0, -1},   //
      Eigen::Vector2i{1, -1}    //
  };

  while (!q.empty())
  {
    const auto& p = q.front();
    visited(p) = 2;  // 2 = visited

    if (!is_edgel(p))
      throw std::runtime_error{"NOT AN EDGEL!"};

    // Find its corresponding node in the disjoint set.
    const auto node_p = ds.node(index(p));

    // Add nonvisited weak edges.
    for (const auto& d : dir)
    {
      const Eigen::Vector2i n = p + d;
      // Boundary conditions.
      if (n.x() < 0 || n.x() >= edges.width() ||  //
          n.y() < 0 || n.y() >= edges.height())
        continue;

      // Make sure that the neighbor is an edgel.
      if (!is_edgel(n))
        continue;

      const auto& comp_p = ds.component(index(n));
      const auto& comp_n = ds.component(index(n));
      const auto& up = vec(statistics[comp_p].angle());
      const auto& un = vec(statistics[comp_n].angle());

      // Merge component of p and component of n if angularly consistent.
      if (angular_distance(up, un) < angular_threshold)
      {
        const auto node_n = ds.node(index(n));
        ds.join(node_p, node_n);

        // Update the component statistics.
        const auto& comp_pn = ds.component(index(n));
        if (comp_pn == comp_p)
        {
          statistics[comp_pn].total_mass += statistics[comp_n].total_mass;

          statistics[comp_pn].orientation_sum +=
              statistics[comp_n].orientation_sum;
          statistics[comp_pn].unnormalized_center +=
              statistics[comp_n].unnormalized_center;
          statistics[comp_pn].unnormalized_inertia +=
              statistics[comp_n].unnormalized_inertia;
        }
        else  // if (comp_pn == comp_n)
        {
          statistics[comp_pn].total_mass += statistics[comp_n].total_mass;
          statistics[comp_pn].orientation_sum +=
              statistics[comp_p].orientation_sum;
          statistics[comp_pn].unnormalized_center +=
              statistics[comp_p].unnormalized_center;
          statistics[comp_pn].unnormalized_inertia +=
              statistics[comp_p].unnormalized_inertia;
        }
      }

      // Enqueue the neighbor n if it is not already enqueued
      if (visited(n) == 0)
      {
        // Enqueue the neighbor.
        q.emplace(n);
        visited(n) = 1;  // 1 = enqueued
      }
    }

    q.pop();
  }

  auto contours = std::map<int, std::vector<Point2i>>{};
  for (auto y = 0; y < edges.height(); ++y)
  {
    for (auto x = 0; x < edges.width(); ++x)
    {
      const auto p = Eigen::Vector2i{x, y};
      const auto index_p = index(p);
      if (is_edgel(p))
        contours[static_cast<int>(ds.component(index_p))].push_back(p);
    }
  }

  return contours;
}

}
