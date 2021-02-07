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
  auto find_dominant_vanishing_point(const TensorView_<T, 2>& lines,
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


  template <typename T>
  struct DominantDirectionSolver3D
  {
    using model_type = Eigen::Matrix<T, 3, 1>;

    static constexpr auto num_points = 1;

    template <typename BackProjectedPlane>
    inline auto operator()(const BackProjectedPlane& plane) const -> model_type
    {
      return plane.transpose().template block<3, 1>(0, 0);
    }
  };

  template <typename T>
  struct AngularDistance3D
  {
    using model_type = Eigen::Matrix<T, 3, 1>;
    using scalar_type = T;

    AngularDistance3D() = default;

    AngularDistance3D(const model_type& n) noexcept
      : normal{n}
    {
    }

    template <typename Mat>
    inline auto operator()(const Mat& planes_backprojected) const
        -> Eigen::Matrix<T, Eigen::Dynamic, 1>
    {
      auto distances = Eigen::Matrix<T, Eigen::Dynamic, 1>(planes_backprojected.rows());
      for (auto i = 0; i < planes_backprojected.rows(); ++i)
      {
        model_type plane_normal = planes_backprojected.row(i).transpose().head(3);
        model_type n_times_v = plane_normal.cross(normal);
        distances(i) = n_times_v.norm();
      }

      return distances;
    }

    Eigen::Matrix<T, 3, 1> normal;
  };

  template <typename T>
  auto find_dominant_direction(const TensorView_<T, 2>& planes,
                                      T angle_threshold,
                                      std::size_t num_random_samples = 100)
  {
    auto vp_solver = DominantDirectionSolver3D<T>{};
    auto inlier_predicate = InlierPredicate<AngularDistance3D<T>>{
        {},                        //
        std::sin(angle_threshold)  //
    };
    return ransac(planes,            //
                  vp_solver,         //
                  inlier_predicate,  //
                  num_random_samples);
  }


  template <typename T>
  struct DominantOrthgonalDirectionTripletSolver3D
  {
    // A rotation matrix.
    using model_type = Eigen::Matrix<T, 3, 3>;

    static constexpr auto num_points = 3;

    template <typename BackProjectedPlane>
    inline auto operator()(const BackProjectedPlane& plane_triplet) const -> model_type
    {
      const auto normals = plane_triplet.transpose().template block<3, 3>(0, 0);
      auto R = model_type{};
      R.col(0) = normals.col(0).cross(normals.col(1));
      R.col(1) = R.col(0).cross(normals.col(2)).normalized();
      R.col(2) = R.col(0).cross(R.col(1));
      // std::cout << "planes =\n" << plane_normals << std::endl << std::endl;
      // std::cout << "R_cand =\n" << R << std::endl << std::endl;
      return R;
    }
  };

  template <typename T>
  struct InvarianceAngularDistance3D
  {
    using model_type = Eigen::Matrix<T, 3, 3>;
    using scalar_type = T;

    InvarianceAngularDistance3D() = default;

    InvarianceAngularDistance3D(const model_type& r) noexcept
      : rotation{r}
    {
    }

    template <typename Mat>
    inline auto operator()(const Mat& planes_backprojected) const
        -> Eigen::Matrix<T, Eigen::Dynamic, 1>
    {
      auto distances = Eigen::Matrix<T, Eigen::Dynamic, 1>(planes_backprojected.rows());

      for (auto i = 0; i < planes_backprojected.rows(); ++i)
      {
        const Eigen::Matrix<T, 3, 1> normal = planes_backprojected.row(i).transpose().head(3);
        const Eigen::Matrix<T, 3, 1> plane_normal_rotated = rotation * plane_normal;
        distances(i) = plane_normal_rotated.cross(plane_normal).norm();
      }

      return distances;
    }

    Eigen::Matrix<T, 3, 3> rotation;
  };

  template <typename T>
  auto find_dominant_orthogonal_direction_triplet(
      const TensorView_<T, 2>& planes, T angle_threshold,
      std::size_t num_random_samples = 100)
  {
    auto vp_solver = DominantOrthgonalDirectionTripletSolver3D<T>{};
    auto inlier_predicate = InlierPredicate<InvarianceAngularDistance3D<T>>{
        {},                        //
        std::sin(angle_threshold)  //
    };
    return ransac(planes,            //
                  vp_solver,         //
                  inlier_predicate,  //
                  num_random_samples);
  }

}  // namespace DO::Sara
