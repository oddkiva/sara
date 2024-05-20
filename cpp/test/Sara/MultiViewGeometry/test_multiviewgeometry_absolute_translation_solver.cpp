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

#define BOOST_TEST_MODULE                                                      \
  "MultiViewGeometry/Minimal Solvers/Absolute Translation"

#include "SyntheticDataUtilities.hpp"

#include <DO/Sara/Core/EigenFormatInterop.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/AbsoluteTranslationSolver.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_absolute_translation_solver)
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

    // Apply the global rotation.
    const Eigen::MatrixXd Xc0 = C.R * Xw.colwise().hnormalized();

    auto Yc = Xc.topRows<3>().colwise().normalized();
    std::cout << "* Backprojected rays:" << std::endl;
    std::cout << "  Yc =\n" << Yc << std::endl;
    std::cout << "* Yc column norms " << std::endl;
    std::cout << "  column_norm(Yc) = " << Yc.colwise().norm() << std::endl;

    // Rx + t = sy
    // -t + sy = Rx
    const auto tsolver = sara::AbsoluteTranslationSolver<double>{};
    const Eigen::Matrix<double, 3, 2> x = Xc0.leftCols<2>();
    const Eigen::Matrix<double, 3, 2> y = Yc.leftCols<2>();
    const auto [t, s] = tsolver(x, y);

    fmt::print("t_est = {}\n", t.transpose().eval());
    fmt::print("t_gdt = {}\n", C.t.transpose().eval());

    BOOST_CHECK_SMALL((t - C.t).norm() / C.t.norm(), 1e-5);
    BOOST_CHECK((s.array() > 0).all());
  }
}
