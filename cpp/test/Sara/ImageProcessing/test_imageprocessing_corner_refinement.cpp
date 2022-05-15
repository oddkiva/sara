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

#define BOOST_TEST_MODULE "ImageProcessing/Forstner's Corner Refinement Method"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/CornerRefinement.hpp>
#include <DO/Sara/ImageProcessing/Differential.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_corner_refinement)
{
  auto f = sara::Image<float>{9, 9};
  // clang-format off
  f.matrix() <<
  //0  1  2  3  4
    0, 0, 0, 0, 0, 0, 0, 0, 0, // 0
    0, 0, 0, 0, 0, 0, 0, 0, 0, // 1
    0, 0, 0, 0, 0, 0, 0, 0, 0, // 2
    0, 0, 0, 0, 0, 0, 0, 0, 0, // 3
    0, 0, 0, 0, 0, 0, 0, 0, 0, // 4
    1, 1, 1, 1, 0, 0, 0, 0, 0, // 5
    1, 1, 1, 1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 0, 0, 0, 0, 0;
  // clang-format on

  const auto grad_f = sara::gradient(f);

  auto grad_f_x = sara::Image<float>{f.sizes()};
  auto grad_f_y = sara::Image<float>{f.sizes()};
  std::transform(grad_f.begin(), grad_f.end(), grad_f_x.begin(),
                 [](const auto& gradient) { return gradient.x(); });
  std::transform(grad_f.begin(), grad_f.end(), grad_f_y.begin(),
                 [](const auto& gradient) { return gradient.y(); });


  std::cout << f.matrix() << std::endl;
  std::cout << grad_f_x.matrix() << std::endl;
  std::cout << grad_f_y.matrix() << std::endl;

  const auto corner = Eigen::Vector2i{5, 3};
  const auto corner_refined =
      sara::refine_corner_location_unsafe(grad_f, corner, 3);
  std::cout << "refined_corner = " << corner_refined.transpose() << std::endl;
  BOOST_CHECK_CLOSE(corner_refined.x(), 3.5, 1e-6);
  BOOST_CHECK_CLOSE(corner_refined.y(), 4.5, 1e-6);
}
