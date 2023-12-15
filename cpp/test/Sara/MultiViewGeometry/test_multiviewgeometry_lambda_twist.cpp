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

#define BOOST_TEST_MODULE "MultiViewGeometry/Perspective-n-Point/Lambda-Twist"

#include "SyntheticDataUtilities.hpp"

#include <DO/Sara/MultiViewGeometry/PnP/LambdaTwist.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_lambda_twist)
{
  const auto xa = std::array{0.0, 0.1, 0.3, 0.0};
  const auto ya = std::array{0.0, 0.2, 0.2, 0.1};
  const auto za = std::array{0.0, 0.3, 0.1, 0.0};
  auto Xw = make_cube_vertices();

  // Get the test camera matrix.
  for (auto i = 0u; i < xa.size(); ++i)
  {
    const auto C = make_camera(xa[i], ya[i], za[i]);

    auto Xc = to_camera_coordinates(C, Xw);
    std::cout << "* Camera Coordinates:" << std::endl;
    std::cout << "  Xc =\n" << Xc << std::endl;

    auto Yc = Xc.topRows<3>().colwise().normalized();
    std::cout << "* Backprojected Light Rays:" << std::endl;
    std::cout << "  Yc =\n" << Yc << std::endl;
    std::cout << "* Yc column norms " << std::endl;
    std::cout << "  column_norm(Yc) = " << Yc.colwise().norm() << std::endl;
    BOOST_REQUIRE_SMALL(
        (Yc.colwise().norm() - Eigen::MatrixXd::Ones(1, 8)).norm(), 1e-12  //
    );

    auto lt = sara::LambdaTwist<double>{
        Xw.topLeftCorner<3, 3>(),  //
        Yc.leftCols<3>()           //
    };

    SARA_DEBUG << "X = scene points =\n" << lt.x << std::endl;
    SARA_DEBUG << "Y = rays =\n" << lt.y << std::endl;
    SARA_DEBUG << "colwise norm(y) = " << lt.y.colwise().squaredNorm() << std::endl;

    // -------------------------------------------------------------------------
    // Calculate the auxiliary variables.
    // -------------------------------------------------------------------------
    // We expect the user to normalize the 3D rays before feeding the data to
    // Lambda-Twist, so let us check that first.
    sara::print_stage("Auxiliary variables");
    lt.calculate_auxiliary_variables();
    SARA_DEBUG << "a = squared distances = " << lt.a.transpose() << std::endl;
    SARA_DEBUG << "b = angle cosines between rays = " << lt.b.transpose() << std::endl;
    BOOST_REQUIRE_SMALL(
        (lt.y.colwise().norm() - Eigen::RowVector3d::Ones()).norm(),
        1e-12
    );

    // By definition, the inhomogeneous quadrics M[_01], M[_02], M[_03] are
    // cylindric ellipses.
    //
    // Let us print the inhomogeneous quadrics.
    SARA_DEBUG << "M01 =\n" << lt.M[0] << std::endl;
    SARA_DEBUG << "M02 =\n" << lt.M[1] << std::endl;
    SARA_DEBUG << "M12 =\n" << lt.M[2] << std::endl;
    // Let us print the two homogeneous quadrics.
    SARA_DEBUG << "D0 =\n" << lt.D[0] << std::endl;
    SARA_DEBUG << "D1 =\n" << lt.D[1] << std::endl;
    // Let us check the cylindric property for each of three inhomogeneous
    // quadrics.
    for (auto k = 0; k < 3; ++k)
      BOOST_REQUIRE_EQUAL(lt.M[k].determinant(), 0);
    // Check that these three inhomogeneous quadrics are also cylindric ellipses.
    const auto M01_planar = lt.M[0].topLeftCorner<2, 2>();
    const auto M12_planar = lt.M[2].bottomRightCorner<2, 2>();
    BOOST_REQUIRE(M01_planar.determinant() > 0);
    BOOST_REQUIRE(M12_planar.determinant() > 0);
    // TODO: I am lazy to check for lt.M[1].


    // We know the following squared distances between the first 3
    // vertices because the data consists of cube vertices,
    BOOST_REQUIRE_SMALL((lt.a - Eigen::Vector3d{1, 1, 2}).norm(), 1e-12);
    // The angles between the 3 vertices is 90 degrees, because again they are
    // cube vertices.
    BOOST_REQUIRE_SMALL((lt.b - Eigen::Vector3d::Ones()).norm(), 2e-2);


    // -------------------------------------------------------------------------
    // Solve the cubic equation.
    // -------------------------------------------------------------------------
    sara::print_stage("Solve cubic polynomial");
    lt.solve_cubic_equation();
    SARA_CHECK(lt.gamma[0]);
    SARA_CHECK(lt.gamma[1]);
    SARA_CHECK(lt.gamma[2]);
    SARA_CHECK(lt.c(lt.gamma[0]));
    // We only care about the first root. Our implementation makes sure that
    // the first is always well formed.
    BOOST_REQUIRE_SMALL(lt.c(lt.gamma[0]), 1.2e-12);
    // Form the linear combination of the two homogenous quadrics.
    const Eigen::Matrix3d D0 = lt.D[0] + lt.gamma[0] * lt.D[1];
    // The goal of the cubic equation was to ensure its determinant is zero.
    BOOST_REQUIRE_SMALL(D0.determinant(), 1e-10);


    // -------------------------------------------------------------------------
    // Solve for the scales λ.
    // -------------------------------------------------------------------------
    sara::print_stage("solve for lambda");
    lt.solve_for_lambda();
    // Check the eigenvalue decomposition of the linear combination D0.
    //
    // First the span of eigenvectors should be right-handed and orthonormal.
    BOOST_REQUIRE_SMALL(
        (lt.E.colwise().norm() - Eigen::RowVector3d::Ones()).norm(), 1e-12);
    BOOST_REQUIRE_CLOSE(lt.E.determinant(), 1, 1e-12);
    // Second the eigenvalues should be ordered as in the paper:
    // - the first one > 0
    // - the second one < 0
    // - the third one = 0
    BOOST_REQUIRE_GT(lt.sigma(0), 0);
    BOOST_REQUIRE_LT(lt.sigma(1), 0);
    BOOST_REQUIRE_SMALL(lt.sigma(2), 1e-12);
    // Finally let us ensure that the eigendecomposition is quantitatively
    // correct.
    BOOST_REQUIRE_SMALL(
        (D0 - lt.E * lt.sigma.asDiagonal() * lt.E.transpose()).norm(), 1e-12);

    // Check each candidate scale vector λ[k] that they belong in the quadrics.
    for (const auto& lambda: lt.lambda_k) {
      // Every scale must be positive to be physically feasible.
      BOOST_REQUIRE((lambda.array() > 0).all());

      // The scale vector λ must be in the two homogeneous quadrics.
      for (auto k = 0; k < 2; ++k)
      {
        const double val = lambda.transpose() * lt.D[k] * lambda;
        BOOST_REQUIRE_SMALL(val, 1e-12);
      }

      // The scale vector λ must be in the three inhomogeneous quadrics.
      for (auto k = 0; k < 3; ++k)
      {
        const double val = lambda.transpose() * lt.M[k] * lambda;
        BOOST_REQUIRE_CLOSE(val, lt.a(k), 1e-6);
      }
    }

    // -------------------------------------------------------------------------
    // Recover the poses from the candidate scale vector λ[k].
    // -------------------------------------------------------------------------
    sara::print_stage("Recover poses");
    lt.recover_all_poses();
    // Check that there is at least one correct pose.
    BOOST_REQUIRE(std::any_of(  //
        std::begin(lt.pose_k), std::end(lt.pose_k),
        [&C](const Eigen::Matrix<double, 3, 4>& pose) {
          SARA_DEBUG << "[GT] pose =\n" << C.matrix() << std::endl;
          SARA_DEBUG << "[US] pose =\n" << pose << std::endl;
          return (C.matrix() - pose).norm() < 1e-6;
        })  //
    );
    // Make sure the rotational part of each pose is indeed a rotation.
    for (const auto& pose: lt.pose_k) {
      const auto R = pose.topLeftCorner<3, 3>();
      BOOST_REQUIRE_CLOSE(R.determinant(), 1, 1e-6);
    }

    // Finally test again with the function that wraps it all.
    const auto candidate_poses = sara::solve_p3p(  //
        Xw.topLeftCorner<3, 3>().eval(),           //
        Yc.leftCols<3>().eval()                    //
    );
    BOOST_REQUIRE(std::any_of(  //
        std::begin(candidate_poses), std::end(candidate_poses),
        [&C](const Eigen::Matrix<double, 3, 4>& pose) {
          SARA_DEBUG << "[GT] pose =\n" << C.matrix() << std::endl;
          SARA_DEBUG << "[US] pose =\n" << pose << std::endl;
          return (C.matrix() - pose).norm() < 1e-6;
        })  //
    );
  }
}
