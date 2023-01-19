// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>


namespace DO::Sara {

  template <typename T>
  struct VanishingPointSolver
  {
    using model_type = Projective::Point2<T>;
    using data_point_type = TensorView_<T, 2>;

    static constexpr auto num_points = 2;
    static constexpr auto num_models = 1;

    inline auto operator()(const data_point_type& ab) const
        -> std::array<model_type, num_models>
    {
      const auto abT = ab.colmajor_view().matrix();
      const Projective::Point2<T> a = abT.col(0);
      const Projective::Point2<T> b = abT.col(1);

      auto p = Projective::intersection(a, b);
      p /= p(2);
      return {p};
    }
  };

  template <typename T>
  struct LineToVanishingPointDistance
  {
    using model_type = Projective::Point2<T>;
    using scalar_type = T;

    LineToVanishingPointDistance() = default;

    auto set_model(const Projective::Point2<T>& p) noexcept
    {
      vp = p;
    }

    template <typename Derived>
    inline auto operator()(const Eigen::MatrixBase<Derived>& lines) const
        -> Eigen::RowVector<T, Eigen::Dynamic>
    {
      return (vp.transpose() * lines).cwiseAbs();
    }

    Projective::Point2<T> vp;
  };


  template <typename T>
  struct DominantOrthogonalDirectionTripletSolver3D
  {
    // A rotation matrix.
    using model_type = Eigen::Matrix3<T>;
    using data_point_type = TensorView_<T, 2>;

    static constexpr auto debug = false;
    static constexpr auto num_points = 3;
    static constexpr auto num_models = 1;

    inline auto operator()(const data_point_type& plane_triplet) const
        -> std::array<model_type, num_models>
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
      const Eigen::Matrix3<T> plane_normals =
          plane_triplet.colmajor_view().matrix().template block<3, 3>(0, 0);
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

      return {R};
    }
  };

  template <typename T>
  struct AngularDistance3D
  {
    using model_type = Eigen::Matrix3<T>;
    using scalar_type = T;

    AngularDistance3D() = default;

    auto set_model(const model_type& r) noexcept
    {
      rotation = r;
    }

    template <typename Derived>
    inline auto
    operator()(const Eigen::MatrixBase<Derived>& planes_backprojected) const
        -> Eigen::Vector<T, Eigen::Dynamic>
    {
      auto distances = Eigen::Vector<T, Eigen::Dynamic>(  //
          planes_backprojected.cols()                     //
      );

      // Check whether a plane contains a vanishing point/direction.
      distances = (rotation * planes_backprojected.topRows(3))
                      .cwiseAbs()
                      .colwise()
                      .minCoeff();

      return distances;
    }

    Eigen::Matrix3<T> rotation =
        Eigen::Matrix3<T>::Constant(std::numeric_limits<T>::quiet_NaN());
  };


  template <typename T>
  auto find_dominant_vanishing_point(const PointList<T, 2>& lines,
                                     T threshold = 5.f /* pixels */,
                                     std::size_t num_random_samples = 100)
  {
    auto vp_solver = VanishingPointSolver<T>{};
    auto inlier_predicate = InlierPredicate<LineToVanishingPointDistance<T>>{
        {}, threshold  //
    };
    return ransac(lines,             //
                  vp_solver,         //
                  inlier_predicate,  //
                  num_random_samples);
  }

  template <typename T>
  auto find_dominant_orthogonal_directions(const PointList<T, 2>& planes,  //
                                           T threshold,                    //
                                           std::size_t num_random_samples = 100)
  {
    auto vp_solver = DominantOrthogonalDirectionTripletSolver3D<T>{};
    auto inlier_predicate = InlierPredicate<AngularDistance3D<T>>{
        {}, threshold  //
    };
    return ransac(planes,            //
                  vp_solver,         //
                  inlier_predicate,  //
                  num_random_samples);
  }

}  // namespace DO::Sara
