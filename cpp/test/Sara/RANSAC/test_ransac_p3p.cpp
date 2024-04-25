// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "RANSAC/P3P Solver"

#include "../MultiViewGeometry/SyntheticDataUtilities.hpp"

#include <DO/Sara/Core/EigenFormatInterop.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/P3PSolver.hpp>
#include <DO/Sara/RANSAC/RANSACv2.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_ransac_with_p3p_solver)
{
  // The scene points are cube vertices.
  //
  // The cube is at the origin of the world frame.
  // The cube is about 10 meters away of the camera center.
  const auto Xw = make_cube_vertices();

  // Scene point Euclidean coordinates in the world frame.
  const Eigen::MatrixXd Xwe = Xw.colwise().hnormalized();


  // Generate some small rotations for the camera.
  const auto xa = std::array{0.0, 0.1, 0.3, 0.0};
  const auto ya = std::array{0.0, 0.2, 0.2, 0.1};
  const auto za = std::array{0.0, 0.3, 0.1, 0.0};

  auto camera = sara::v2::PinholeCamera<double>{};
  camera.focal_lengths() << 1000, 1000;
  camera.principal_point() << 960, 540;
  camera.shear() = 0;


  // The machinery for the P3P solver.
  const auto p3p_solver = sara::P3PSolver<double>{};
  auto p3p_inlier_predicate = sara::CheiralPnPConsistency<  //
      sara::v2::PinholeCamera<double>>{};
  p3p_inlier_predicate.set_camera(camera);
  p3p_inlier_predicate.ε = 0.2 /* pixels */;


  for (auto i = 0u; i < xa.size(); ++i)
  {
    const auto C_gt = make_camera(xa[i], ya[i], za[i]);

    auto Xc = to_camera_coordinates(C_gt, Xw);

    // Scene point Euclidean coordinates in the camera frame.
    const Eigen::MatrixXd Xce = Xc.colwise().hnormalized();
    // Check that the scene points are all in front of the camera.
    BOOST_REQUIRE((Xce.row(2).array() > 0).all());

    // The backprojected rays coordinates in the camera frame.
    const Eigen::MatrixXd Yc = Xc.topRows<3>().colwise().normalized();
    BOOST_REQUIRE_SMALL(
        (Yc.colwise().norm() - Eigen::MatrixXd::Ones(1, 8)).norm(), 1e-12  //
    );
    // We do a dummy test for now: we use the perfectly backprojected rays.


#define CHECK_P3P_SOLVER_AND_CHEIRAL_P3P_CONSISTENCY
#if defined(CHECK_P3P_SOLVER_AND_CHEIRAL_P3P_CONSISTENCY)
    // Project the scene points to the image plane.
    fmt::print("* Image Coordinates:\n");
    auto u = Eigen::MatrixXd{2, Xw.cols()};
    for (auto j = 0; j < Xw.cols(); ++j)
      u.col(j) = camera.project(Xce.col(j));
    fmt::print("  u =\n{}\n", u);

    // Intermediate data transformations for the P3P solver.
    //
    // 1. The first 3 world scene points.
    auto Xwe_tensor = sara::TensorView_<double, 2>{
        const_cast<double*>(Xwe.data()),  //
        {3, 3}                            //
    };
    // 2. The corresponding backprojected rays.
    const auto Yc_tensor = sara::TensorView_<double, 2>{
        const_cast<double*>(Yc.data()),  //
        {3, 3}                           //
    };
    BOOST_REQUIRE(Xwe.leftCols(3) == Xwe_tensor.colmajor_view().matrix());
    BOOST_REQUIRE(Yc.leftCols(3) == Yc_tensor.colmajor_view().matrix());

    const auto C_est = p3p_solver(Xwe_tensor, Yc_tensor);
    BOOST_REQUIRE(!C_est.empty());

    auto errs = std::vector<double>(C_est.size());
    for (auto p = 0u; p < C_est.size(); ++p)
    {
      errs[p] = (C_est[p] - C_gt.matrix()).norm() / C_gt.matrix().norm();
      fmt::print("* Pose GT =\n{}\n", C_gt.matrix());
      fmt::print("* Pose ES[{}] =\n{}\n", p, C_est[p]);
      fmt::print("* err[{}] = {}\n", p, errs[p]);

      const Eigen::MatrixXd Xc_est = C_est[p] * Xw;
      fmt::print("* Camera coordinates estimated\n");
      fmt::print("  Xc_est =\n{}\n", Xc_est);

      auto u_est = Eigen::MatrixXd{2, Xw.cols()};
      for (auto c = 0; c < Xc_est.cols(); ++c)
        u_est.col(c) = camera.project(Xc_est.col(c).eval());

      const Eigen::MatrixXd err_mat = u_est - u;
      fmt::print("u_est - u =\n{}\n", err_mat);

      p3p_inlier_predicate.set_model(C_est[p]);

      auto M = sara::PointRayCorrespondenceList<double>{};
      M.x = sara::TensorView_<double, 2>{const_cast<double*>(Xw.data()),
                                         {Xw.cols(), Xw.rows()}};
      M.y = sara::TensorView_<double, 2>{const_cast<double*>(Yc.data()),
                                         {Yc.cols(), Yc.rows()}};
      const auto inliers = p3p_inlier_predicate(M).count();
      fmt::print("inlier count = {}\n", inliers);
    }
#endif

    fmt::print("RUNNING RANSAC\n");
    auto point_ray_pairs = sara::PointRayCorrespondenceList<double>{};
    point_ray_pairs.x = sara::TensorView_<double, 2>{
        const_cast<double*>(Xw.data()), {Xw.cols(), Xw.rows()}};
    point_ray_pairs.y = sara::TensorView_<double, 2>{
        const_cast<double*>(Yc.data()), {Yc.cols(), Yc.rows()}};
    const auto [pose, inliers, sample_best] = sara::v2::ransac(
        point_ray_pairs, p3p_solver, p3p_inlier_predicate, 10, 0.99);

    // The estimation should be perfect.
    BOOST_REQUIRE(inliers.flat_array().count() == Xw.cols());

    const auto pose_err = (pose - C_gt.matrix()).norm() / C_gt.matrix().norm();
    BOOST_REQUIRE_SMALL(pose_err, 1e-12);
  }
}
