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
#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/Triangulation.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;


auto generate_test_data()
{
  // 3D points.
  auto X = Eigen::MatrixXd(4, 5);  // coefficients are in [-1, 1].
  // clang-format off
  X.topRows<3>() <<
    -1.49998, -0.5827,   -1.40591,   0.369386,  0.161931,
    -1.23692, -0.434466, -0.142271, -0.732996, -1.43086,
     1.51121,  0.437918,  1.35859,   1.03883,   0.106923;
  // clang-format on
  X.bottomRows<1>().fill(1.);

  const auto R = rotation(0.3, 0.2, 0.1);
  const auto t = Eigen::Vector3d{0.1, 0.2, 0.3};

  const auto E = essential_matrix(R, t);

  const Matrix34d C1 =
      normalized_camera(Matrix3d::Identity(), Vector3d::Zero()).matrix();
  const Matrix34d C2 = normalized_camera(R, t).matrix();
  MatrixXd x1 = C1 * X;
  x1.array().rowwise() /= x1.row(2).array();
  MatrixXd x2 = C2 * X;
  x2.array().rowwise() /= x2.row(2).array();

  return std::make_tuple(X, R, t, E, C1, C2, x1, x2);
}

template <typename T>
auto triangulate_nister(const Eigen::Matrix3<T>& E,
                        const Eigen::Matrix<T, 3, 4>& P,
                        const Eigen::Vector3<T>& ray1,
                        const Eigen::Vector3<T>& ray2) -> Eigen::Vector4<T>
{
  const Eigen::Vector3<T> a = E.transpose() * ray2;
  const Eigen::Vector3<T> b =
      ray1.cross(Eigen::Vector3<T>(1, 1, 0).asDiagonal() * a);
  const Eigen::Vector3<T> c =
      ray2.cross(Eigen::Vector3<T>(1, 1, 0).asDiagonal() * E * ray1);
  const Eigen::Vector3<T> d = a.cross(b);
  const Eigen::Vector4<T> C = P.transpose() * c;

  auto X = Eigen::Vector4<T>{};
  X << d * C(3), -d.dot(C.head(3));
  return X;
}

BOOST_AUTO_TEST_CASE(test_triangulate_linear_eigen_v2)
{
  const auto [X, R, t, E, C1, C2, x1, x2] = generate_test_data();
  (void) R;
  (void) t;
  (void) E;

  const auto [X_est, s1, s2] = triangulate_linear_eigen(C1, C2, x1, x2);

  BOOST_CHECK_SMALL((X - X_est).norm() / X.norm(), 1e-6);
  // The proper cheirality check is the following for general camera models.
  //
  // For camera with angles > 180 degrees, pixels can be imaged from light rays
  // behind the camera.
  BOOST_CHECK((s1.array() > 0).all());
  BOOST_CHECK((s2.array() > 0).all());

  SARA_DEBUG << "C1 * X_est = " << std::endl << C1 * X_est << std::endl;
  SARA_DEBUG << "s1 * x1 = " << std::endl
             << x1.array().rowwise() * s1.transpose().array() << std::endl;
  SARA_DEBUG << "C2 * X_est = " << std::endl << C1 * X_est << std::endl;
  SARA_DEBUG << "s2 * x2 = " << std::endl
             << x1.array().rowwise() * s1.transpose().array() << std::endl;

  const auto s1_x1 = (x1.array().rowwise() * s1.transpose().array()).matrix();
  const auto s2_x2 = (x2.array().rowwise() * s2.transpose().array()).matrix();

  BOOST_CHECK_SMALL((C1 * X_est - s1_x1.matrix()).norm(), 1e-6);
  BOOST_CHECK_SMALL((C2 * X_est - s2_x2).norm(), 1e-6);
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
    const auto z1 = X.colwise().hnormalized().row(2);
    BOOST_CHECK((z1.array() > 0).all());

    // Cheirality with respect to P2 = [R|t].
    const Matrix34d P2 = normalized_camera(*motion_found);
    const Eigen::MatrixXd X2 = (P2 * X);
    const auto z2 = X2.row(2);
    BOOST_CHECK((z2.array() > 0).all());

    auto [X_est, s1, s2] = triangulate_linear_eigen(P1, P2, x1, x2);
    BOOST_CHECK((X_est.row(2).array() > 0).all());
    BOOST_CHECK((s1.array() > 0).all());
    BOOST_CHECK((s2.array() > 0).all());
  }

  for (auto motion = candidate_motions.begin();
       motion != candidate_motions.end(); ++motion)
  {
    if (motion_found == motion)
      continue;

    const Matrix34d P2_est = normalized_camera(motion->R, motion->t);

    auto [X_est, s1, s2] = triangulate_linear_eigen(P1, P2_est, x1, x2);
    BOOST_CHECK_LT((s1.array() > 0 && s2.array() > 0).count(), 5);

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
    std::cout << std::endl;

    SARA_DEBUG << "In front of camera P2 = "
               << ((P2_est * X_est).row(2).array() > 0) << std::endl;
    SARA_DEBUG << "All in front of camera P2 = "
               << ((P2_est * X_est).row(2).array() > 0).all() << std::endl;
    SARA_DEBUG << "Count in front of camera P2 = "
               << ((P2_est * X_est).row(2).array() > 0).count() << std::endl;

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
  (void) X;
  (void) E;
  (void) P1;
  (void) P2;

  const auto g = two_view_geometry(Motion{R, t}, x1, x2);
  BOOST_CHECK(g.cheirality.all());
  BOOST_CHECK_EQUAL(g.C1.matrix(), normalized_camera().matrix());
  BOOST_CHECK_EQUAL(g.C2.matrix(),
                    normalized_camera(R, t.normalized()).matrix());
}

BOOST_AUTO_TEST_CASE(test_triangulate_nister)
{
  const auto [X, R, t, E, C1, C2, x1, x2] = generate_test_data();
  (void) R;
  (void) t;
  (void) E;

  const auto P = normalized_camera(R, t).matrix();
  SARA_DEBUG << "P =\n" << P << std::endl;
  for (auto i = 0; i < x1.cols(); ++i)
  {
    const Eigen::Vector3d x1i = x1.col(i);
    const Eigen::Vector3d x2i = x1.col(i);
    const auto Xi = triangulate_nister(E.matrix(), P, x1i, x2i);
    SARA_CHECK(i);
    SARA_CHECK(X.col(i).transpose());
    SARA_CHECK(Xi.hnormalized().homogeneous().transpose());
  }
}
