#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Sara/Geometry/Algorithms/RobustEstimation/RANSAC.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>


namespace DO::Sara {

  template <typename T>
  struct VanishingPointSolver
  {
    using model_type = Projective::Line2<T>;

    static constexpr auto num_points = 2;

    template <typename Mat>
    inline auto operator()(const Mat& ab) const
    {
      const Eigen::Matrix<T, 3, 2> abT = ab.transpose();
      const auto& a = abT.col(0);
      const auto& b = abT.col(1);

      auto p = Projective::intersection(a.eval(), b.eval());
      p /= p(2);
      return p;
    }
  };

  template <typename T>
  struct LineToVanishingPointDistance
  {
    using model_type = Projective::Point2<T>;
    using scalar_type = T;

    LineToVanishingPointDistance() = default;

    LineToVanishingPointDistance(const Projective::Point2<T>& p) noexcept
      : vp{p}
    {
    }

    template <typename Mat>
    inline auto operator()(const Mat& lines) const
        -> Eigen::Matrix<T, Eigen::Dynamic, 1>
    {
      return (lines * vp).cwiseAbs();
    }

    Projective::Point2<T> vp;
  };

  template <typename T>
  auto find_dominant_vanishing_point(const Tensor_<T, 2>& lines,
                                     T threshold = 5.f /* pixels */,
                                     std::size_t num_random_samples = 100)
  {
    auto vp_solver = VanishingPointSolver<T>{};
    auto inlier_predicate = InlierPredicate<LineToVanishingPointDistance<T>>{
        {},        //
        threshold  //
    };
    return ransac(lines,             //
                  vp_solver,         //
                  inlier_predicate,  //
                  num_random_samples);
  }


}  // namespace DO::Sara
