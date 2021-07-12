// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Match/IndexMatch Data Structures"

#include <DO/Sara/Match/HDF5.hpp>
#include <DO/Sara/Match/IndexMatch.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_hdf5_read_write_index_match)
{
  const auto filepath = (fs::temp_directory_path() / "test.h5").string();

  // Write data to HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_TRUNC};

    auto matches = std::vector<IndexMatch>{
        // K, R, t.
        {0, 1, 0.f},
        {1, 2, 1.f},
    };
    h5file.write_dataset("matches", tensor_view(matches));
  }

  // Read data from from HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_RDONLY};
    auto matches = std::vector<IndexMatch>{};
    h5file.read_dataset("matches", matches);

    const auto& m0 = matches[0];
    const auto& m1 = matches[1];

    BOOST_CHECK_EQUAL(m0.i, 0);
    BOOST_CHECK_EQUAL(m0.j, 1);
    BOOST_CHECK_EQUAL(m0.score, 0.f);

    BOOST_CHECK_EQUAL(m1.i, 1);
    BOOST_CHECK_EQUAL(m1.j, 2);
    BOOST_CHECK_EQUAL(m1.score, 1.f);
  }

  fs::remove(filepath);
}
