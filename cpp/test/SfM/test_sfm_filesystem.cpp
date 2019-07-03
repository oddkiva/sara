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


#define BOOST_TEST_MODULE "SfM/Filesytem Utilities"

#include <DO/Sara/FileSystem.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;
using namespace std;


BOOST_AUTO_TEST_SUITE(TestSfMFileSystem)

BOOST_AUTO_TEST_CASE(test_)
{
  auto images = sara::ls(
      "/mnt/a1cc5981-3655-4f74-9c62-37253d79c82d/sfm/Alamo/images", ".jpg");
}

BOOST_AUTO_TEST_SUITE_END()
