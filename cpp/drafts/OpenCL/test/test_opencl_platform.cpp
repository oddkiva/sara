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

#include <drafts/OpenCL/CommandQueue.hpp>
#include <drafts/OpenCL/Context.hpp>
#include <drafts/OpenCL/Device.hpp>
#include <drafts/OpenCL/DeviceBuffer.hpp>
#include <drafts/OpenCL/Kernel.hpp>
#include <drafts/OpenCL/Platform.hpp>
#include <drafts/OpenCL/Program.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_get_platforms)
{
  vector<Platform> platforms_list = get_platforms();
  BOOST_CHECK(!platforms_list.empty());

  for (const auto& platform : platforms_list)
  {
    BOOST_CHECK(platform.id != nullptr);
    BOOST_CHECK(!platform.name.empty());
    BOOST_CHECK(!platform.vendor.empty());
    BOOST_CHECK(!platform.version.empty());
    BOOST_CHECK(!platform.profile.empty());
    BOOST_CHECK(!platform.extensions.empty());
  }
}
