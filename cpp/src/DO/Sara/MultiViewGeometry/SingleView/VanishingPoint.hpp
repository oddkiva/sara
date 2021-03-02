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
  struct DominantOrthogonalDirectionTripletSolver3D
  {
    // A rotation matrix.
    using matrix_type = Eigen::Matrix<T, 3, 3>;
    using model_type = Eigen::Matrix<T, 3, 3>;

    static constexpr auto debug = false;
    static constexpr auto num_points = 3;

    template <typename BackProjectedPlane>
    inline auto operator()(const BackProjectedPlane& plane_triplet) const -> model_type
    {
      // Each image line backprojects to a plane passing through the camera
      // center.
      //
      // Our matrix data stores plane normals as row vectors.
      //
      // So we preprocess the data by:
      // 1. Transposing the matrix
      // 2. Keeping the first three rows
      // to obtain the normals of backprojected planes as column vectors.
      const matrix_type plane_normals =
          plane_triplet.transpose().template block<3, 3>(0, 0);
      if (debug)
      {
        std::cout << "Plane triplet dimensions" << std::endl;
        std::cout << plane_triplet.rows() << "x" << plane_triplet.cols()
                  << std::endl;
      }
      const auto n0 = plane_normals.col(0);
      const auto n1 = plane_normals.col(1);
      const auto n2 = plane_normals.col(2);

      // Calculate the principal orthogonal directions.
      auto R = model_type{};
      auto v0 = R.col(0);
      auto v1 = R.col(1);
      auto v2 = R.col(2);

      // Vanishing points are 3D rays passing through the camera centers.
      //
      // First vanishing point as a 3D ray.
      //
      // If the two image lines intersect in a vanishing point, then that means
      // that their corresponding backprojected planes both contains the
      // backprojected direction vector.
      //
      // This intersecting direction can be computed as the cross-product of the
      // two plane normals:
      v0 = n0.cross(n1).normalized();

      // The ray backprojected from the first vanishing point can be viewed as a
      // normal vector of a plane.
      //
      // There are two key observations:
      // 1. The second (and third orthogonal) direction backprojected from the
      //    second (orthogonal) VP must lie in this plane.
      // 2. If the third image line vanishes into this second VP, then that
      //    means that its backprojected plane contains this backprojected
      //    direction.
      //
      // The backprojected direction is at the intersection of the two planes
      // orthogonal, in other words it can calculated by the cross-product
      v1 = v0.cross(n2).normalized();

      // Third vanishing point is straightforwardly calculated as:
      v2 = v0.cross(v1).normalized();

      return R;
    }
  };

  template <typename T>
  struct AngularDistance3D
  {
    using model_type = Eigen::Matrix<T, 3, 3>;
    using scalar_type = T;

    AngularDistance3D() = default;

    AngularDistance3D(const model_type& r) noexcept
      : rotation{r}
    {
    }

    template <typename Mat>
    inline auto operator()(const Mat& planes_backprojected) const
        -> Eigen::Matrix<T, Eigen::Dynamic, 1>
    {
      auto distances = Eigen::Matrix<T, Eigen::Dynamic, 1>(  //
          planes_backprojected.rows()                        //
      );

      // Check whether a plane contains a vanishing point/direction.
      distances = (planes_backprojected.leftCols(3) * rotation)
                      .cwiseAbs()
                      .rowwise()
                      .minCoeff();

      return distances;
    }

    Eigen::Matrix<T, 3, 3> rotation;
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
  auto find_dominant_orthogonal_directions(
      const TensorView_<T, 2>& planes,  //
      T threshold,                      //
      std::size_t num_random_samples = 100)
  {
    auto vp_solver = DominantOrthogonalDirectionTripletSolver3D<T>{};
    auto inlier_predicate = InlierPredicate<AngularDistance3D<T>>{
        {},        //
        threshold  //
    };
    return ransac(planes,            //
                  vp_solver,         //
                  inlier_predicate,  //
                  num_random_samples);
  }

}  // namespace DO::Sara
