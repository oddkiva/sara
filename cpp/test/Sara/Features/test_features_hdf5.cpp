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

#define BOOST_TEST_MODULE "HDF5 I/O"

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Features/IO.hpp>

#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <iostream>


namespace fs = std::filesystem;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_hdf5_read_write_array)
{
  const auto filepath = (fs::current_path() / "sfm_dummy_data.h5").string();

  // Write.
  {
    auto h5file = H5File{filepath, H5F_ACC_TRUNC};

    const auto group = h5file.get_group("0");

    // Create dummy features.
    auto features = Tensor_<OERegion, 1>{4};
    auto farray = features.flat_array();
    for (int i = 0; i < 4; ++i)
    {
      farray(i).center() << float(i), float(i);
      farray(i).shape_matrix = Eigen::Matrix2f::Ones() * (i + 0.5f);
      farray(i).orientation = 30.f * i;
      farray(i).extremum_value = 10.f * i;
    }

    auto dataset = h5file.write_dataset("0/features", features);
  }

  // Read.
  {
    auto h5file = H5File{filepath, H5F_ACC_RDONLY};

    auto features = Tensor_<OERegion, 1>{};
    h5file.read_dataset("0/features", features);

    auto farray = features.flat_array();

    for (int i = 0; i < farray.size(); ++i)
    {
      BOOST_CHECK_EQUAL(farray(i).center(), Point2f::Ones() * i);
      BOOST_CHECK_EQUAL(farray(i).shape_matrix,
                        Eigen::Matrix2f::Ones() * (i + 0.5f));
      BOOST_CHECK_EQUAL(farray(i).orientation, 30 * i);
      BOOST_CHECK_EQUAL(farray(i).extremum_value, 10 * i);
    }
  }
}
