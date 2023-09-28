// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "OpenCL/Core Utilities"

#include <boost/test/unit_test.hpp>

#include <DO/Shakti/OpenCL/Device.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_device)
{
  Platform platform = get_platforms().front();
  BOOST_CHECK(!get_devices(platform).empty());
}
