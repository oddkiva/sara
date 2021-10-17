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

    // We expect the user to normalize the 3D rays before feeding the data to
    // Lambda-Twist, so let us check that first.
    lt.calculate_auxiliary_variables();
    BOOST_REQUIRE_SMALL(
        (lt.y.colwise().norm() - Eigen::RowVector3d::Ones()).norm(),
        1e-12
    );

    // By definition, the homogeneous quadrics M[_01], M[_02], M[_03] are
    // cylindric ellipses.
    //
    // Let us check the cylindric properties.
    for (auto i = 0; i < 3; ++i)
      BOOST_REQUIRE_EQUAL(lt.M[i].determinant(), 0);
    // Check that these three cylindric quadrics are cylindric ellipses.
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
  }
}
