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
  auto x = Placeholder<Tensor<float, 4>>('x');
  auto conv2d = Conv2D{};
}

BOOST_AUTO_TEST_SUITE_END()
