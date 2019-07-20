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

#define BOOST_TEST_MODULE "Filesytem Utilities"

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/FileSystem.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;
namespace sara = DO::Sara;


BOOST_AUTO_TEST_SUITE(TestSfMFileSystem)

BOOST_AUTO_TEST_CASE(test_ls)
{
  {
    auto images = sara::ls(fs::temp_directory_path().string(), ".jpg");
    BOOST_CHECK(images.empty());
  }

  {
    auto images = sara::ls(src_path("../../../../data/"), ".jpg");
    BOOST_CHECK(!images.empty());
  }
}

BOOST_AUTO_TEST_SUITE_END()
