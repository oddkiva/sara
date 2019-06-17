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

#define BOOST_TEST_MODULE "NeuralNetworks/Layers"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/NeuralNetworks/Layers.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_SUITE(TestLayers)

BOOST_AUTO_TEST_CASE(test_conv2d)
{
  constexpr auto N = 1;
  constexpr auto H = 4;
  constexpr auto W = 3;
  constexpr auto C = 3;

  auto x = Tensor_<float, 4>{};
  {
    x = Tensor_<float, 4>{{N, H, W, C}};
    x.flat_array() <<
      0,0,0,   1, 1, 1,   2, 2, 2,
      3,3,3,   4, 4, 4,   5, 5, 5,
      6,6,6,   7, 7, 7,   8, 8, 8,
      9,9,9,  10,10,10,  11,11,11;

    for (int i = 1; i < N; ++i)
      x.flat_array() = x[0].flat_array();
  }

  auto conv2d = Conv2D{};
  {
    constexpr auto kH = 3;
    constexpr auto kW = 3;
    constexpr auto kC = 3;
    auto kt = Tensor_<float, 4>{{kC, kH, kW, kC}};
    {
      auto ktr = kt.reshape(Vector2i{kC * kH * kW, kC});
      //                   kC x kH x kW  kO
      ktr.matrix().col(0) << VectorXf::Ones(9), VectorXf::Zero(9),
          VectorXf::Zero(9);
      ktr.matrix().col(1) << VectorXf::Zero(9), VectorXf::Ones(9),
          VectorXf::Zero(9);
      ktr.matrix().col(2) << VectorXf::Zero(9), VectorXf::Zero(9),
          VectorXf::Ones(9);
    };

    conv2d.w = kt;
  }

  auto y = conv2d(x);
}

BOOST_AUTO_TEST_SUITE_END()
