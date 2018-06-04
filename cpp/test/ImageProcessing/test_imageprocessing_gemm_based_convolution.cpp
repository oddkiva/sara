// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/GEMM-based Convolution"

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>

#include <boost/test/unit_test.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_im2col)
{
  constexpr auto N = 3;
  constexpr auto H = 4;
  constexpr auto W = 3;
  auto x = Tensor_<float, 3>{{N, H, W}};

  auto plane = Tensor_<float, 2>{{H, W}};
  plane.flat_array() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9,10,11;

  x.flat_array() <<            //
      1 * plane.flat_array(),  //
      2 * plane.flat_array(),  //
      3 * plane.flat_array();

  std::cout << "x[0].matrix() = \n" << x[0].matrix() << std::endl;
  std::cout << "x[1].matrix() = \n" << x[1].matrix() << std::endl;
  std::cout << "x[2].matrix() = \n" << x[2].matrix() << std::endl;

  constexpr auto kH = 3;
  constexpr auto kW = 3;

  auto phi_x = Tensor_<float, 2>{{N * H * W, kH * kW}};
  auto phi_x_0 = im2col(x[0], {kH, kW});
  auto phi_x_1 = im2col(x[1], {kH, kW});
  auto phi_x_2 = im2col(x[2], {kH, kW});

  //BOOST_CHECK(phi_x.sizes() == Vector2i(N * H * W, kH * kW));

  //auto true_phi_x = MatrixXf{N*H*W, kH*kW};
  //true_phi_x <<
  //  // (0, 0)
  //  // 0, 0, 0,
  //  // 0, 0, 0,
  //  // 0, 0, 0,
  //  // (0, 1)
  //  // 0, 0, 0,
  //  // 0, 0, 0,
  //  // 0, 0, 0,
  //  // (0, 2)
  //  // 0, 0, 0,
  //  // 0, 0, 0,
  //  // 0, 0, 0,
  //  //
  //  // And so on.
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,
  //  0, 0, 0,   0, 0, 0,   0, 0, 0,

  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,
  //  1, 1, 1,   1, 1, 1,   1, 1, 1,

  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2,
  //  2, 2, 2,   2, 2, 2,   2, 2, 2;

  //BOOST_CHECK(phi_x.matrix() == true_phi_x);
}

BOOST_AUTO_TEST_CASE(test_convolve)
{
  constexpr auto N = 10;
  constexpr auto H = 4;
  constexpr auto W = 6;
  constexpr auto C = 3;
  auto x = Tensor_<float, 4>{{N, C, H, W}};
  x.flat_array().fill(1.f);

  constexpr auto kN = 5;
  constexpr auto kH = 3;
  constexpr auto kW = 3;
  constexpr auto kC = 3;
  auto k = Tensor_<float, 4>{{kN, kC, kH, kW}};

  k.flat_array() <<
    // R
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    // G
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    // B
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,

    // R
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,
    // G
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,
    // B
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,

    // R
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,
    // G
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,
    // B
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,

    // R
    4, 4, 4,
    4, 4, 4,
    4, 4, 4,
    // G
    4, 4, 4,
    4, 4, 4,
    4, 4, 4,
    // B
    4, 4, 4,
    4, 4, 4,
    4, 4, 4,

    // R
    5, 5, 5,
    5, 5, 5,
    5, 5, 5,
    // G
    5, 5, 5,
    5, 5, 5,
    5, 5, 5,
    // B
    5, 5, 5,
    5, 5, 5,
    5, 5, 5;

  auto y = x;
  gemm_convolve(y, x, k);
}
