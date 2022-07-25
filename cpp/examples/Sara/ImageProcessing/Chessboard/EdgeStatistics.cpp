#include "EdgeStatistics.hpp"

namespace sara = DO::Sara;

auto get_curve_shape_statistics(
    const std::vector<std::vector<Eigen::Vector2i>>& curve_pts)
    -> sara::CurveStatistics
{
  auto curves_64f = std::vector<std::vector<Eigen::Vector2d>>{};
  curves_64f.resize(curve_pts.size());
  std::transform(curve_pts.begin(), curve_pts.end(), curves_64f.begin(),
                 [](const auto& points) {
                   auto points_2d = std::vector<Eigen::Vector2d>{};
                   points_2d.resize(points.size());
                   std::transform(
                       points.begin(), points.end(), points_2d.begin(),
                       [](const auto& p) { return p.template cast<double>(); });
                   return points_2d;
                 });

  return sara::CurveStatistics{curves_64f};
}

auto mean_gradient(
    const std::vector<std::vector<Eigen::Vector2i>>& curve_pts,  //
    const sara::ImageView<float>& Ix,                            //
    const sara::ImageView<float>& Iy)                            //
    -> std::vector<Eigen::Vector2f>
{
  auto mean_gradients = std::vector<Eigen::Vector2f>(curve_pts.size());

  std::transform(
      curve_pts.begin(), curve_pts.end(), mean_gradients.begin(),
      [&Ix, &Iy](const std::vector<Eigen::Vector2i>& points) {
        static const Eigen::Vector2f zero2f = Eigen::Vector2f::Zero();
        Eigen::Vector2f g = std::accumulate(
            points.begin(), points.end(), zero2f,
            [&Ix, &Iy](const Eigen::Vector2f& gradient,
                       const Eigen::Vector2i& point) -> Eigen::Vector2f {
              const Eigen::Vector2f g =
                  gradient + Eigen::Vector2f{Ix(point), Iy(point)};
              return g;
            });
        const Eigen::Vector2f mean = g / points.size();
        return mean;
      });

  return mean_gradients;
}
