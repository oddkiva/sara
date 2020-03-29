// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Halide Backend/SeparableConvolution2d"

#include <DO/Sara/Core/Tensor.hpp>

#include <drafts/Halide/Helpers.hpp>
#include <drafts/Halide/Utilities.hpp>

#include <boost/test/unit_test.hpp>

#include "SeparableConvolution2d.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


using namespace sara;


BOOST_AUTO_TEST_CASE(test_separable_convolution_2d)
{
}
