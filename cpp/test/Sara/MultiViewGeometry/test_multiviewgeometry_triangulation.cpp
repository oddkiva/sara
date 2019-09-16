// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "MultiViewGeometry/Triangulation"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/Triangulation.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;


auto generate_test_data()
{
  // 3D points.
  MatrixXd X(4, 5);  // coefficients are in [-1, 1].
  X.topRows<3>() <<
    -1.49998,   -0.5827,  -1.40591,  0.369386,  0.161931, //
    -1.23692, -0.434466, -0.142271, -0.732996,  -1.43086, //
     1.51121,  0.437918,   1.35859,   1.03883,  0.106923; //
  X.bottomRows<1>().fill(1.);

  const Matrix3d R = rotation_z(0.3) * rotation_x(0.1) * rotation_y(0.2);
  const Vector3d t{0.1, 0.2, 0.3};

  const auto E = essential_matrix(R, t);

  const Matrix34d C1 = PinholeCamera{Matrix3d::Identity(), Matrix3d::Identity(),
                                     Vector3d::Zero()};
  const Matrix34d C2 = PinholeCamera{Matrix3d::Identity(), R, t};
  MatrixXd x1 = C1 * X; x1.array().rowwise() /= x1.row(2).array();
  MatrixXd x2 = C2 * X; x2.array().rowwise() /= x2.row(2).array();

  return std::make_tuple(X, R, t, E, C1, C2, x1, x2);
}


BOOST_AUTO_TEST_CASE(test_triangulate_linear_eigen)
{
  const auto [X, R, t, E, C1, C2, x1, x2] = generate_test_data();

  MatrixXd X_est = triangulate_linear_eigen(C1, C2, x1, x2);

  BOOST_CHECK_SMALL((X - X_est).norm() / X.norm(), 1e-6);
}


BOOST_AUTO_TEST_CASE(test_cheirality_predicate)
{
  const auto [X, R, t, E, P1, P2, x1, x2] = generate_test_data();

  const auto candidate_motions = extract_relative_motion_horn(E);

  const auto true_motion = Motion{R, t};

  const auto thres = 1e-12;
  auto motion_equality_predicate = [&](const auto& motion) {
    const auto rotation_equality =
        (motion.R - true_motion.R).norm() / true_motion.R.norm() < thres;
    const auto translation_equality =
        (motion.t.normalized() - true_motion.t.normalized()).norm() /
            true_motion.t.normalized().norm() <
        thres;
    return rotation_equality && translation_equality;
  };

  const auto motion_found =
      std::find_if(candidate_motions.begin(), candidate_motions.end(),
                   motion_equality_predicate);
  BOOST_CHECK(motion_found != candidate_motions.end());
  SARA_DEBUG << "Motion found" << std::endl;
  SARA_DEBUG << "R =\n" << motion_found->R << std::endl;
  SARA_DEBUG << "t = " << motion_found->t.normalized().transpose() << std::endl;
  SARA_DEBUG << "ΔR = " << (motion_found->R - true_motion.R).norm()
             << std::endl;
  SARA_DEBUG
      << "Δt = "
      << (motion_found->t.normalized() - true_motion.t.normalized()).norm()
      << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  // Check the motion is completely cheiral w.r.t. to all the 5 point
  // correspondences.
  {
    // Cheirality with respect to P1 = [I|0].
    BOOST_CHECK_EQUAL(cheirality_predicate(X.colwise().hnormalized()).count(),
                      5);
    BOOST_CHECK_EQUAL(cheirality_predicate(X).count(), 5);
    // Cheirality with respect to P2 = [R|t].
    const Matrix34d P2 = normalized_camera(*motion_found);
    BOOST_CHECK_EQUAL(cheirality_predicate(P2 * X).count(), 5);
    BOOST_CHECK_EQUAL(relative_motion_cheirality_predicate(X, P2).count(), 5);
  }

  for (auto motion = candidate_motions.begin();
       motion != candidate_motions.end(); ++motion)
  {
    if (motion_found == motion)
      continue;

    const Matrix34d P2_est = normalized_camera(motion->R, motion->t);
    BOOST_CHECK(relative_motion_cheirality_predicate(X, P2_est).count() < 5);

    auto X_est = triangulate_linear_eigen(P1, P2_est, x1, x2);

    SARA_DEBUG << "candidate camera =" << std::endl;
    std::cout << P2_est << std::endl;

    SARA_DEBUG << "true camera =" << std::endl;
    std::cout << P2 << std::endl;
    SARA_DEBUG << "ΔR = " << (motion->R - R).norm() << std::endl;
    SARA_DEBUG << "Δt = " << (motion->t.normalized() - t.normalized()).norm()
               << std::endl;
    SARA_DEBUG << "ΔX = " << (X - X_est).norm() / X.norm() << std::endl;
    std::cout << std::endl;

    SARA_DEBUG << "X_est =" << std::endl;
    std::cout << X_est << std::endl;

    SARA_DEBUG << "P2_est * X_est =" << std::endl;
    std::cout << (P2_est * X_est) << std::endl;
    std::cout << std::endl;

    SARA_DEBUG << "In front of camera P1 = " << (X_est.row(2).array() > 0)
               << std::endl;
    SARA_DEBUG << "All in front of camera P1 = "
               << (X_est.row(2).array() > 0).all() << std::endl;
    SARA_DEBUG << "Count in front of camera P1 = "
               << (X_est.row(2).array() > 0).count() << std::endl;
    SARA_DEBUG << "cheirality_check = " << cheirality_predicate(X_est)
               << std::endl;
    std::cout << std::endl;

    SARA_DEBUG << "In front of camera P2 = "
               << ((P2_est * X_est).row(2).array() > 0) << std::endl;
    SARA_DEBUG << "All in front of camera P2 = "
               << ((P2_est * X_est).row(2).array() > 0).all() << std::endl;
    SARA_DEBUG << "Count in front of camera P2 = "
               << ((P2_est * X_est).row(2).array() > 0).count() << std::endl;
    SARA_DEBUG << "cheirality_check = " << cheirality_predicate(P2_est * X_est)
               << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
  };


  SARA_DEBUG << "Filtering based on cheirality constraints..." << std::endl;

  // Check the cheirality filter.
  auto geometries = std::vector<TwoViewGeometry>{};
  std::transform(std::begin(candidate_motions), std::end(candidate_motions),
                 std::back_inserter(geometries),
                 [&, x1 = std::cref(x1), x2 = std::cref(x2)](const Motion& m) {
                   return two_view_geometry(m, x1, x2);
                 });
  geometries.erase(std::remove_if(std::begin(geometries), std::end(geometries),
                                  [&, X = X](const auto& g) {
                                    return g.cheirality.count() != X.cols();
                                  }),
                   std::end(geometries));

  BOOST_CHECK_EQUAL(geometries.size(), 1u);

  const auto& g = geometries.front();
  const auto& C2 = g.C2;
  BOOST_CHECK(motion_equality_predicate(Motion{C2.R, C2.t}));

  SARA_DEBUG << "X =" << std::endl;
  std::cout << X << std::endl;

  const auto& X_est = g.X;
  SARA_DEBUG << "X_est =" << std::endl;
  std::cout << X_est << std::endl;

  // Check the ratio of coordinates, it must be constant.
  const ArrayXXd ratio =
      X.colwise().hnormalized().array() / X_est.colwise().hnormalized().array();
  SARA_DEBUG << "Ratio =" << std::endl;
  std::cout << ratio << std::endl;

  const auto min_ratio = ratio.minCoeff();
  const auto max_ratio = ratio.minCoeff();
  BOOST_CHECK(min_ratio > 0);
  BOOST_CHECK(max_ratio > 0);

  const auto rel_ratio_diff = (max_ratio - min_ratio) / max_ratio;
  BOOST_CHECK_SMALL(rel_ratio_diff, 1e-12);
  SARA_CHECK(rel_ratio_diff);
}

BOOST_AUTO_TEST_CASE(test_calculate_two_view_geometries)
{
  const auto [X, R, t, E, P1, P2, x1, x2] = generate_test_data();

  const auto g = two_view_geometry(Motion{R, t}, x1, x2);
  BOOST_CHECK(g.cheirality.all());
  BOOST_CHECK_EQUAL(g.C1.matrix(), normalized_camera().matrix());
  BOOST_CHECK_EQUAL(g.C2.matrix(),
                    normalized_camera(R, t.normalized()).matrix());
}
